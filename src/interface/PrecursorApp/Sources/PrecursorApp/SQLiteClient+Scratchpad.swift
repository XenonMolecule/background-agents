import Foundation
import SQLite3

extension SQLiteClient {
    // MARK: - Scratchpad (Project Memory)

    private func bindOptionalText(_ stmt: OpaquePointer?, _ index: Int32, _ value: String?) {
        if let v = value, !v.isEmpty {
            bindText(stmt, index, v)
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func ensureScratchpadSortOrdersInitialized(projectName: String) throws {
        try open()
        let needsSql = """
        SELECT 1
        FROM scratchpad_entries
        WHERE project_name = ?
          AND status = 'active'
          AND sort_order IS NULL
        LIMIT 1
        """
        var needsStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, needsSql, -1, &needsStmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare ensureScratchpadSortOrdersInitialized(needs)")
        }
        defer { sqlite3_finalize(needsStmt) }
        bindText(needsStmt, 1, projectName)
        let needs = (sqlite3_step(needsStmt) == SQLITE_ROW)
        if !needs { return }

        // Group by (section, subsection).
        let groupsSql = """
        SELECT DISTINCT section, COALESCE(subsection, '') AS subsection_norm
        FROM scratchpad_entries
        WHERE project_name = ?
          AND status = 'active'
        """
        var groupsStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, groupsSql, -1, &groupsStmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare ensureScratchpadSortOrdersInitialized(groups)")
        }
        defer { sqlite3_finalize(groupsStmt) }
        bindText(groupsStmt, 1, projectName)

        // Prepare update statement once.
        let updateSql = "UPDATE scratchpad_entries SET sort_order = ? WHERE id = ?"
        var updateStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, updateSql, -1, &updateStmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare ensureScratchpadSortOrdersInitialized(update)")
        }
        defer { sqlite3_finalize(updateStmt) }

        try exec("BEGIN IMMEDIATE TRANSACTION;")
        defer { try? exec("COMMIT;") }

        while sqlite3_step(groupsStmt) == SQLITE_ROW {
            let section = String(cString: sqlite3_column_text(groupsStmt, 0))
            let subNorm = String(cString: sqlite3_column_text(groupsStmt, 1))

            let idsSql: String
            if subNorm.isEmpty {
                idsSql = """
                SELECT id
                FROM scratchpad_entries
                WHERE project_name = ?
                  AND section = ?
                  AND subsection IS NULL
                  AND status = 'active'
                ORDER BY datetime(created_at) ASC, id ASC
                """
            } else {
                idsSql = """
                SELECT id
                FROM scratchpad_entries
                WHERE project_name = ?
                  AND section = ?
                  AND COALESCE(subsection, '') = ?
                  AND status = 'active'
                ORDER BY datetime(created_at) ASC, id ASC
                """
            }

            var idsStmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, idsSql, -1, &idsStmt, nil) == SQLITE_OK else {
                throw sqliteError("prepare ensureScratchpadSortOrdersInitialized(ids)")
            }
            defer { sqlite3_finalize(idsStmt) }
            bindText(idsStmt, 1, projectName)
            bindText(idsStmt, 2, section)
            if !subNorm.isEmpty {
                bindText(idsStmt, 3, subNorm)
            }

            var idx: Int32 = 0
            while sqlite3_step(idsStmt) == SQLITE_ROW {
                let id = sqlite3_column_int64(idsStmt, 0)
                sqlite3_reset(updateStmt)
                sqlite3_clear_bindings(updateStmt)
                sqlite3_bind_int(updateStmt, 1, idx)
                sqlite3_bind_int64(updateStmt, 2, id)
                guard sqlite3_step(updateStmt) == SQLITE_DONE else {
                    throw sqliteError("step ensureScratchpadSortOrdersInitialized(update)")
                }
                idx += 1
            }
        }
    }

    func listScratchpadEntries(projectName: String) throws -> [ScratchpadEntry] {
        try open()
        try ensureScratchpadSortOrdersInitialized(projectName: projectName)

        let sql = """
        SELECT id, project_name, section, subsection, message, confidence,
               COALESCE(sort_order, 0) AS sort_order,
               COALESCE(last_edited_by, 'system') AS last_edited_by,
               created_at
        FROM scratchpad_entries
        WHERE status = 'active'
          AND project_name = ?
        ORDER BY section COLLATE NOCASE ASC, sort_order ASC, datetime(created_at) ASC, id ASC
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare listScratchpadEntries")
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, projectName)

        var out: [ScratchpadEntry] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let id = sqlite3_column_int64(stmt, 0)
            let proj = String(cString: sqlite3_column_text(stmt, 1))
            let section = String(cString: sqlite3_column_text(stmt, 2))
            let subsection: String? = sqlite3_column_text(stmt, 3).map { String(cString: $0) }
            let message = String(cString: sqlite3_column_text(stmt, 4))
            let confidence = Int(sqlite3_column_int64(stmt, 5))
            let sortOrder = Int(sqlite3_column_int64(stmt, 6))
            let lastEditedBy = String(cString: sqlite3_column_text(stmt, 7))
            let createdAtStr = String(cString: sqlite3_column_text(stmt, 8))
            let createdAt = DateFormatter.sqlite.date(from: createdAtStr) ?? Date()

            out.append(
                ScratchpadEntry(
                    id: id,
                    projectName: proj,
                    section: section,
                    subsection: subsection,
                    message: message,
                    confidence: confidence,
                    sortOrder: sortOrder,
                    lastEditedBy: lastEditedBy,
                    createdAt: createdAt
                )
            )
        }
        return out
    }

    func addScratchpadEntry(
        projectName: String,
        section: String,
        subsection: String?,
        message: String,
        confidence: Int
    ) throws -> Int64 {
        try open()
        try ensureScratchpadSortOrdersInitialized(projectName: projectName)

        let maxSql = """
        SELECT COALESCE(MAX(sort_order), -1) + 1
        FROM scratchpad_entries
        WHERE status = 'active'
          AND project_name = ?
          AND section = ?
          AND COALESCE(subsection, '') = COALESCE(?, '')
        """
        var maxStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, maxSql, -1, &maxStmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare addScratchpadEntry(max)")
        }
        defer { sqlite3_finalize(maxStmt) }
        bindText(maxStmt, 1, projectName)
        bindText(maxStmt, 2, section)
        bindOptionalText(maxStmt, 3, subsection)
        guard sqlite3_step(maxStmt) == SQLITE_ROW else {
            throw sqliteError("step addScratchpadEntry(max)")
        }
        let nextSort = Int(sqlite3_column_int64(maxStmt, 0))

        let sql = """
        INSERT INTO scratchpad_entries (project_name, section, subsection, message, confidence, sort_order, last_edited_by, status)
        VALUES (?, ?, ?, ?, ?, ?, 'user', 'active')
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare addScratchpadEntry")
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, projectName)
        bindText(stmt, 2, section)
        bindOptionalText(stmt, 3, subsection)
        bindText(stmt, 4, message)
        sqlite3_bind_int(stmt, 5, Int32(max(0, min(10, confidence))))
        sqlite3_bind_int(stmt, 6, Int32(max(0, nextSort)))

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw sqliteError("step addScratchpadEntry")
        }
        return sqlite3_last_insert_rowid(db)
    }

    func updateScratchpadEntry(id: Int64, message: String, confidence: Int) throws {
        try open()
        let sql = """
        UPDATE scratchpad_entries
        SET message = ?, confidence = ?, last_edited_by = 'user'
        WHERE id = ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare updateScratchpadEntry")
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, message)
        sqlite3_bind_int(stmt, 2, Int32(max(0, min(10, confidence))))
        sqlite3_bind_int64(stmt, 3, id)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw sqliteError("step updateScratchpadEntry")
        }
    }

    func deleteScratchpadEntry(id: Int64) throws {
        try open()
        let sql = """
        UPDATE scratchpad_entries
        SET status = 'deleted', last_edited_by = 'user'
        WHERE id = ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare deleteScratchpadEntry")
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int64(stmt, 1, id)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw sqliteError("step deleteScratchpadEntry")
        }
    }

    func reorderScratchpadEntries(
        projectName: String,
        section: String,
        subsection: String?,
        orderedIds: [Int64]
    ) throws {
        try open()
        try ensureScratchpadSortOrdersInitialized(projectName: projectName)

        let sql = """
        UPDATE scratchpad_entries
        SET sort_order = ?, last_edited_by = 'user'
        WHERE id = ?
          AND project_name = ?
          AND section = ?
          AND COALESCE(subsection, '') = COALESCE(?, '')
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare reorderScratchpadEntries")
        }
        defer { sqlite3_finalize(stmt) }

        try exec("BEGIN IMMEDIATE TRANSACTION;")
        defer { try? exec("COMMIT;") }

        for (idx, id) in orderedIds.enumerated() {
            sqlite3_reset(stmt)
            sqlite3_clear_bindings(stmt)
            sqlite3_bind_int(stmt, 1, Int32(idx))
            sqlite3_bind_int64(stmt, 2, id)
            bindText(stmt, 3, projectName)
            bindText(stmt, 4, section)
            bindOptionalText(stmt, 5, subsection)
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw sqliteError("step reorderScratchpadEntries")
            }
        }
    }
}


