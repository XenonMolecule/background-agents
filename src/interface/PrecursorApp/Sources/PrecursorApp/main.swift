import SwiftUI
import AppKit
import Foundation
import SQLite3

// MARK: - Models

enum TaskStatus: String {
    case pending = "Agent Completed Tasks (Pending Review)"
    case accepted = "Accepted Agent Completed Tasks"
}

struct AgentTaskItem: Identifiable, Equatable {
    let id: Int64
    let projectName: String
    let status: TaskStatus
    let message: String
    let createdAt: Date
    let metadata: [String: Any]

    static func == (lhs: AgentTaskItem, rhs: AgentTaskItem) -> Bool {
        return lhs.id == rhs.id
    }

    var uri: String? {
        (metadata["uri"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    var taskTitle: String {
        if let t = (metadata["task"] as? String), !t.isEmpty { return t }
        // Fallback: try to extract before " (uri:" if present
        if let idx = message.firstIndex(of: "(") {
            return String(message[..<idx]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return message
    }
    var shortDescription: String? {
        (metadata["short_description"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    var stepByStepSummary: String? {
        (metadata["step_by_step_summary"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

enum ConversationRole: String {
    case user = "user"
    case agent = "agent"
    case system = "system"
}

struct ConversationMessage: Identifiable, Equatable {
    let id: Int64
    let projectName: String
    let role: ConversationRole
    let message: String
    let createdAt: Date
    let seenByUser: Bool
    let visibleToUser: Bool

    static func == (lhs: ConversationMessage, rhs: ConversationMessage) -> Bool {
        lhs.id == rhs.id
    }
}

// MARK: - SQLite Client

final class SQLiteClient {
    // Internal so extensions in other files can implement additional queries.
    var db: OpaquePointer?

    init() {}

    deinit {
        close()
    }

    func open() throws {
        if db != nil { return }
        let path = Self.resolveDatabasePath()
        var handle: OpaquePointer?
        let rc = path.withCString { sqlite3_open($0, &handle) }
        if rc != SQLITE_OK {
            throw NSError(domain: "SQLite", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unable to open database at \(path)"])
        }
        db = handle
        try ensureConversationSchema()
        try ensureScratchpadSchema()
    }

    func close() {
        if let d = db {
            sqlite3_close(d)
            db = nil
        }
    }

    static func resolveDatabasePath() -> String {
        let env = ProcessInfo.processInfo.environment
        if let override = env["PRECURSOR_SCRATCHPAD_DB"], !override.isEmpty {
            return override
        }
        // ~/Library/Application Support/precursor/scratchpad.db
        let urls = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
        let base = urls.first ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        let dir = base.appendingPathComponent("precursor", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let path = dir.appendingPathComponent("scratchpad.db").path
        return path
    }

    func listProjects() throws -> [String] {
        try open()
        let sql = """
        SELECT DISTINCT project_name
        FROM scratchpad_entries
        WHERE status = 'active'
          AND section IN (?, ?)
        ORDER BY project_name COLLATE NOCASE ASC
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare listProjects")
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, TaskStatus.pending.rawValue)
        bindText(stmt, 2, TaskStatus.accepted.rawValue)

        var results: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let cStr = sqlite3_column_text(stmt, 0) {
                results.append(String(cString: cStr))
            }
        }
        return results
    }

    func listTasks(projectName: String) throws -> [AgentTaskItem] {
        try open()
        let sqliteDF: DateFormatter = {
            let df = DateFormatter()
            df.locale = Locale(identifier: "en_US_POSIX")
            df.dateFormat = "yyyy-MM-dd HH:mm:ss"
            // SQLite CURRENT_TIMESTAMP is UTC; interpret stored strings as UTC.
            df.timeZone = TimeZone(secondsFromGMT: 0)
            return df
        }()
        let sql = """
        SELECT id, project_name, section, message, created_at, metadata_json
        FROM scratchpad_entries
        WHERE status = 'active'
          AND project_name = ?
          AND section IN (?, ?)
        ORDER BY datetime(created_at) DESC
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare listTasks")
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, projectName)
        bindText(stmt, 2, TaskStatus.pending.rawValue)
        bindText(stmt, 3, TaskStatus.accepted.rawValue)

        var items: [AgentTaskItem] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let id = sqlite3_column_int64(stmt, 0)
            let proj = String(cString: sqlite3_column_text(stmt, 1))
            let section = String(cString: sqlite3_column_text(stmt, 2))
            let message = String(cString: sqlite3_column_text(stmt, 3))
            let createdAtStr = String(cString: sqlite3_column_text(stmt, 4))
            var metadata: [String: Any] = [:]
            if let metaText = sqlite3_column_text(stmt, 5) {
                let data = Data(String(cString: metaText).utf8)
                if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    metadata = obj
                }
            }
            let status: TaskStatus = (section == TaskStatus.accepted.rawValue) ? .accepted : .pending
            let createdAt = sqliteDF.date(from: createdAtStr) ?? Date()
            let item = AgentTaskItem(
                id: id,
                projectName: proj,
                status: status,
                message: message,
                createdAt: createdAt,
                metadata: metadata
            )
            items.append(item)
        }
        return items
    }

    func updateTaskSection(id: Int64, to newSection: TaskStatus) throws {
        try open()
        let sql = """
        UPDATE scratchpad_entries
        SET section = ?
        WHERE id = ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare updateTaskSection")
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, newSection.rawValue)
        sqlite3_bind_int64(stmt, 2, id)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw sqliteError("step updateTaskSection")
        }
    }

    func updateTaskSectionRaw(id: Int64, to newSection: String) throws {
        try open()
        let sql = """
        UPDATE scratchpad_entries
        SET section = ?
        WHERE id = ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare updateTaskSectionRaw")
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, newSection)
        sqlite3_bind_int64(stmt, 2, id)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw sqliteError("step updateTaskSectionRaw")
        }
    }

    // MARK: - Conversation Messages

    func exec(_ sql: String) throws {
        try open()
        var errMsg: UnsafeMutablePointer<Int8>?
        let rc = sqlite3_exec(db, sql, nil, nil, &errMsg)
        if rc != SQLITE_OK {
            let msg = errMsg.map { String(cString: $0) } ?? "Unknown sqlite error"
            if errMsg != nil { sqlite3_free(errMsg) }
            throw NSError(domain: "SQLite", code: 3, userInfo: [NSLocalizedDescriptionKey: msg])
        }
    }

    func columnExists(table: String, column: String) throws -> Bool {
        try open()
        let sql = "PRAGMA table_info(\(table))"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare pragma table_info")
        }
        defer { sqlite3_finalize(stmt) }
        while sqlite3_step(stmt) == SQLITE_ROW {
            // PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
            if let cStr = sqlite3_column_text(stmt, 1) {
                let name = String(cString: cStr)
                if name == column { return true }
            }
        }
        return false
    }

    private func ensureConversationSchema() throws {
        // Create table if missing (safe to run repeatedly).
        try exec(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                seen_by_user INTEGER NOT NULL DEFAULT 0,
                visible_in_conversation INTEGER NOT NULL DEFAULT 1,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        // Lightweight migration for older DBs that predate visible_in_conversation.
        // (The Python backend also does this, but the UI should be resilient alone.)
        let hasVisible = try columnExists(table: "conversation_messages", column: "visible_in_conversation")
        if !hasVisible {
            try exec(
                "ALTER TABLE conversation_messages ADD COLUMN visible_in_conversation INTEGER NOT NULL DEFAULT 1;"
            )
        }

        // Soft-delete flag: hide messages from both UI + agent context rendering.
        let hasDeleted = try columnExists(table: "conversation_messages", column: "is_deleted")
        if !hasDeleted {
            try exec("ALTER TABLE conversation_messages ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0;")
        }

        // Index for project-scoped fetches.
        try exec(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_project_created
            ON conversation_messages(project_name, created_at);
            """
        )
    }

    // MARK: - Scratchpad schema (shared with Python backend)

    private func ensureScratchpadSchema() throws {
        // Create table if missing (safe to run repeatedly).
        try exec(
            """
            CREATE TABLE IF NOT EXISTS scratchpad_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                section TEXT NOT NULL,
                subsection TEXT,
                message TEXT NOT NULL,
                confidence INTEGER DEFAULT 0,
                sort_order INTEGER,
                last_edited_by TEXT DEFAULT 'system',
                status TEXT DEFAULT 'active',
                metadata_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        // Lightweight migration: add sort_order to older DBs.
        let hasSort = try columnExists(table: "scratchpad_entries", column: "sort_order")
        if !hasSort {
            try exec("ALTER TABLE scratchpad_entries ADD COLUMN sort_order INTEGER;")
        }

        // Lightweight migration: add last_edited_by to older DBs.
        let hasLastEditedBy = try columnExists(table: "scratchpad_entries", column: "last_edited_by")
        if !hasLastEditedBy {
            try exec("ALTER TABLE scratchpad_entries ADD COLUMN last_edited_by TEXT DEFAULT 'system';")
        }

        // Helpful index for project-scoped reads.
        try exec(
            """
            CREATE INDEX IF NOT EXISTS idx_scratchpad_entries_project_section
            ON scratchpad_entries(project_name, section, subsection, sort_order);
            """
        )
    }

    func unreadConversationCountsByProject() throws -> [String: Int] {
        try open()
        let sql = """
        SELECT project_name, COUNT(*) AS c
        FROM conversation_messages
        WHERE role = 'agent'
          AND seen_by_user = 0
          AND visible_in_conversation = 1
          AND is_deleted = 0
        GROUP BY project_name
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare unreadConversationCountsByProject")
        }
        defer { sqlite3_finalize(stmt) }
        var out: [String: Int] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            let proj = String(cString: sqlite3_column_text(stmt, 0))
            let c = Int(sqlite3_column_int64(stmt, 1))
            out[proj] = c
        }
        return out
    }

    func listConversationMessages(projectName: String) throws -> [ConversationMessage] {
        try open()
        let sqliteDF: DateFormatter = {
            let df = DateFormatter()
            df.locale = Locale(identifier: "en_US_POSIX")
            df.dateFormat = "yyyy-MM-dd HH:mm:ss"
            // SQLite CURRENT_TIMESTAMP is UTC; interpret stored strings as UTC.
            df.timeZone = TimeZone(secondsFromGMT: 0)
            return df
        }()
        let sql = """
        SELECT id, project_name, role, message, created_at, seen_by_user, visible_in_conversation
        FROM conversation_messages
        WHERE project_name = ?
          AND visible_in_conversation = 1
          AND is_deleted = 0
        ORDER BY datetime(created_at) ASC, id ASC
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare listConversationMessages")
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, projectName)

        var items: [ConversationMessage] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let id = sqlite3_column_int64(stmt, 0)
            let proj = String(cString: sqlite3_column_text(stmt, 1))
            let roleRaw = String(cString: sqlite3_column_text(stmt, 2))
            let msg = String(cString: sqlite3_column_text(stmt, 3))
            let createdAtStr = String(cString: sqlite3_column_text(stmt, 4))
            let seenByUser = sqlite3_column_int(stmt, 5) != 0
            let visibleToUser = sqlite3_column_int(stmt, 6) != 0

            let role = ConversationRole(rawValue: roleRaw) ?? .agent
            let createdAt = sqliteDF.date(from: createdAtStr) ?? Date()
            items.append(
                ConversationMessage(
                    id: id,
                    projectName: proj,
                    role: role,
                    message: msg,
                    createdAt: createdAt,
                    seenByUser: seenByUser,
                    visibleToUser: visibleToUser
                )
            )
        }
        return items
    }

    func addConversationMessage(
        projectName: String,
        role: ConversationRole,
        message: String,
        seenByUser: Bool,
        visibleToUser: Bool
    ) throws {
        try open()
        let sql = """
        INSERT INTO conversation_messages (project_name, role, message, seen_by_user, visible_in_conversation)
        VALUES (?, ?, ?, ?, ?)
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare addConversationMessage")
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, projectName)
        bindText(stmt, 2, role.rawValue)
        bindText(stmt, 3, message)
        sqlite3_bind_int(stmt, 4, seenByUser ? 1 : 0)
        sqlite3_bind_int(stmt, 5, visibleToUser ? 1 : 0)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw sqliteError("step addConversationMessage")
        }
    }

    func markAgentConversationSeen(projectName: String, upToId: Int64) throws {
        try open()
        let sql = """
        UPDATE conversation_messages
        SET seen_by_user = 1
        WHERE project_name = ?
          AND role = 'agent'
          AND seen_by_user = 0
          AND visible_in_conversation = 1
          AND is_deleted = 0
          AND id <= ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare markAgentConversationSeen")
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, projectName)
        sqlite3_bind_int64(stmt, 2, upToId)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw sqliteError("step markAgentConversationSeen")
        }
    }

    func sqliteError(_ whereMsg: String) -> NSError {
        let errMsg = String(cString: sqlite3_errmsg(db))
        return NSError(domain: "SQLite", code: 2, userInfo: [NSLocalizedDescriptionKey: "\(whereMsg): \(errMsg)"])
    }

    func bindText(_ stmt: OpaquePointer?, _ index: Int32, _ value: String) {
        value.withCString { cStr in
            let sqliteTransient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
            sqlite3_bind_text(stmt, index, cStr, -1, sqliteTransient)
        }
    }
}

extension SQLiteClient {
    func trashConversation(projectName: String) throws {
        try open()
        let sql = """
        UPDATE conversation_messages
        SET is_deleted = 1,
            seen_by_user = 1,
            visible_in_conversation = 0
        WHERE project_name = ?
          AND is_deleted = 0
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError("prepare trashConversation")
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, projectName)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw sqliteError("step trashConversation")
        }
    }
}

extension DateFormatter {
    static let sqlite: DateFormatter = {
        let df = DateFormatter()
        df.locale = Locale(identifier: "en_US_POSIX")
        df.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return df
    }()
}

// MARK: - Grouping

struct TimeBucket: Hashable {
    let title: String
}

private func bucketTitle(for date: Date, now: Date = Date()) -> String {
    let cal = Calendar.current
    if cal.isDateInToday(date) { return "Today" }
    if cal.isDateInYesterday(date) { return "Yesterday" }

    let weekOfYearNow = cal.component(.weekOfYear, from: now)
    let weekOfYearDate = cal.component(.weekOfYear, from: date)
    let yearNow = cal.component(.yearForWeekOfYear, from: now)
    let yearDate = cal.component(.yearForWeekOfYear, from: date)
    if yearNow == yearDate && weekOfYearNow == weekOfYearDate { return "This Week" }
    if yearNow == yearDate && weekOfYearNow == weekOfYearDate + 1 { return "Last Week" }

    let monthNow = cal.component(.month, from: now)
    let monthDate = cal.component(.month, from: date)
    let yearCalNow = cal.component(.year, from: now)
    let yearCalDate = cal.component(.year, from: date)
    if yearCalNow == yearCalDate && monthNow == monthDate { return "This Month" }
    if yearCalNow == yearCalDate && monthNow == monthDate + 1 { return "Last Month" }

    if let sixMonthsAgo = cal.date(byAdding: .month, value: -6, to: now), date >= sixMonthsAgo {
        return "Last 6 Months"
    }
    if cal.component(.year, from: date) == cal.component(.year, from: now) {
        return "This Year"
    }
    return "Last Year"
}

private func groupTasks(_ tasks: [AgentTaskItem]) -> [(String, [AgentTaskItem])] {
    var grouped: [String: [AgentTaskItem]] = [:]
    for t in tasks {
        let title = bucketTitle(for: t.createdAt)
        grouped[title, default: []].append(t)
    }
    // Sort groups by recency, and items within each group by recency desc
    let order = ["Today","Yesterday","This Week","Last Week","This Month","Last Month","Last 6 Months","This Year","Last Year"]
    let sortedKeys = grouped.keys.sorted { a, b in
        let ia = order.firstIndex(of: a) ?? order.count
        let ib = order.firstIndex(of: b) ?? order.count
        if ia != ib { return ia < ib }
        return a < b
    }
    return sortedKeys.map { key in
        let items = grouped[key]?.sorted(by: { $0.createdAt > $1.createdAt }) ?? []
        return (key, items)
    }
}

// MARK: - App State

final class AppState: ObservableObject {
    @Published var projects: [String] = []
    @Published var selectedProject: String? = nil
    @Published var tasks: [AgentTaskItem] = []
    @Published var errorMessage: String? = nil
    @Published var isLoading: Bool = false

    @Published var unreadConversationByProject: [String: Int] = [:]
    @Published var conversationMessages: [ConversationMessage] = []
    @Published var isConversationLoading: Bool = false

    // Internal so other files can extend AppState behavior cleanly.
    let db = SQLiteClient()

    // Scratchpad / Memory
    @Published var scratchpadEntries: [ScratchpadEntry] = []
    @Published var isMemoryLoading: Bool = false
    @Published var memoryErrorMessage: String? = nil

    func loadInitial() {
        isLoading = true
        errorMessage = nil
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let projs = try self.db.listProjects()
                var selected = self.selectedProject
                if selected == nil { selected = projs.first }
                let items = selected != nil ? try self.db.listTasks(projectName: selected!) : []
                let unread = try self.db.unreadConversationCountsByProject()
                DispatchQueue.main.async {
                    self.projects = projs
                    self.selectedProject = selected
                    self.tasks = items
                    self.unreadConversationByProject = unread
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                    self.isLoading = false
                }
            }
        }
    }

    func reloadTasks() {
        guard let project = selectedProject else { return }
        isLoading = true
        errorMessage = nil
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let items = try self.db.listTasks(projectName: project)
                DispatchQueue.main.async {
                    self.tasks = items
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                    self.isLoading = false
                }
            }
        }
    }

    func selectProject(_ project: String) {
        selectedProject = project
        reloadTasks()
    }

    func refreshUnreadConversationCounts() {
        DispatchQueue.global(qos: .utility).async {
            do {
                let unread = try self.db.unreadConversationCountsByProject()
                DispatchQueue.main.async {
                    self.unreadConversationByProject = unread
                }
            } catch {
                // Don't surface as an app-wide error for background polling.
            }
        }
    }

    func loadConversation(projectName: String, markSeen: Bool = true) {
        isConversationLoading = true
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let msgs = try self.db.listConversationMessages(projectName: projectName)
                if markSeen, let lastAgent = msgs.last(where: { $0.role == .agent }) {
                    try self.db.markAgentConversationSeen(projectName: projectName, upToId: lastAgent.id)
                }
                let unread = try self.db.unreadConversationCountsByProject()
                DispatchQueue.main.async {
                    self.conversationMessages = msgs
                    self.unreadConversationByProject = unread
                    self.isConversationLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.isConversationLoading = false
                    self.errorMessage = error.localizedDescription
                }
            }
        }
    }

    func sendUserMessage(projectName: String, text: String, triggerInterviewer: Bool = true) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try self.db.addConversationMessage(
                    projectName: projectName,
                    role: .user,
                    message: trimmed,
                    seenByUser: true,
                    visibleToUser: true
                )
                let msgs = try self.db.listConversationMessages(projectName: projectName)
                DispatchQueue.main.async {
                    self.conversationMessages = msgs
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                }
                return
            }

            guard triggerInterviewer else { return }
            self.runInterviewerCLI(projectName: projectName)
        }
    }

    private func resolveRepoRoot() -> URL? {
        let env = ProcessInfo.processInfo.environment
        if let p = env["PRECURSOR_REPO_ROOT"], !p.isEmpty {
            return URL(fileURLWithPath: p).standardizedFileURL
        }
        let fm = FileManager.default
        var dir = URL(fileURLWithPath: fm.currentDirectoryPath).standardizedFileURL
        for _ in 0..<10 {
            let pyproject = dir.appendingPathComponent("pyproject.toml")
            let precursorDir = dir.appendingPathComponent("src/precursor", isDirectory: true)
            if fm.fileExists(atPath: pyproject.path), fm.fileExists(atPath: precursorDir.path) {
                return dir
            }
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path { break }
            dir = parent
        }
        return nil
    }

    private func runInterviewerCLI(projectName: String) {
        let fm = FileManager.default
        let env = ProcessInfo.processInfo.environment
        var procEnv = env

        // Load settings once; used for python/conda selection.
        let settings: SystemSettingsConfig? = {
            do {
                return try ConfigIO.loadSettings()
            } catch {
                return nil
            }
        }()
        // Env override (debugging) > settings.yaml > default "gum"
        let envOverride = (env["PRECURSOR_CONDA_ENV"]?.trimmingCharacters(in: .whitespacesAndNewlines)).flatMap { $0.isEmpty ? nil : $0 }
        let settingsEnv = settings?.condaEnvName.trimmingCharacters(in: .whitespacesAndNewlines)
        let condaEnvName = envOverride ?? ((settingsEnv?.isEmpty == false) ? settingsEnv! : "gum")

        // Ensure DB path is propagated (so the CLI writes back into the same DB the UI reads).
        procEnv["PRECURSOR_SCRATCHPAD_DB"] = SQLiteClient.resolveDatabasePath()

        let root = resolveRepoRoot()
        if let root {
            let srcPath = root.appendingPathComponent("src").path
            if let existing = procEnv["PYTHONPATH"], !existing.isEmpty {
                procEnv["PYTHONPATH"] = "\(srcPath):\(existing)"
            } else {
                procEnv["PYTHONPATH"] = srcPath
            }
            // Running from repo root makes relative config discovery friendlier.
            procEnv["PRECURSOR_REPO_ROOT"] = root.path
        }

        let p = Process()
        p.environment = procEnv
        p.currentDirectoryURL = root ?? URL(fileURLWithPath: fm.currentDirectoryPath)

        // Capture stdout/stderr for debugging (failures should be surfaced in UI).
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        p.standardOutput = stdoutPipe
        p.standardError = stderrPipe

        func isExecutable(_ path: String) -> Bool {
            fm.isExecutableFile(atPath: path)
        }

        func resolveCondaBin() -> String? {
            if let override = env["PRECURSOR_CONDA_BIN"], !override.isEmpty, isExecutable(override) {
                return override
            }
            let home = NSHomeDirectory()
            let candidates = [
                "/opt/miniconda3/bin/conda",
                "/opt/anaconda3/bin/conda",
                "\(home)/miniconda3/bin/conda",
                "\(home)/anaconda3/bin/conda",
                "\(home)/mambaforge/bin/conda",
                "/opt/homebrew/bin/conda",
            ]
            return candidates.first(where: { isExecutable($0) })
        }

        func resolveSettingsPythonBin() -> String? {
            let raw = (settings?.pythonBin ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if raw.isEmpty { return nil }
            if raw.hasPrefix("./"), let root {
                let p = root.appendingPathComponent(String(raw.dropFirst(2))).path
                return isExecutable(p) ? p : nil
            }
            return isExecutable(raw) ? raw : nil
        }

        func resolveVenvPythonBin() -> String? {
            guard let root else { return nil }
            let p = root.appendingPathComponent(".venv/bin/python").path
            return isExecutable(p) ? p : nil
        }

        // Prefer a fully-qualified python override if provided; then settings python_bin; then .venv; otherwise conda env; then PATH python3.
        if let pythonBin = env["PRECURSOR_PYTHON_BIN"], !pythonBin.isEmpty, isExecutable(pythonBin) {
            p.executableURL = URL(fileURLWithPath: pythonBin)
            p.arguments = [
                "-m",
                "precursor.cli.interviewer_agent_cli",
                "--project",
                projectName,
            ]
        } else if let py = resolveSettingsPythonBin() ?? resolveVenvPythonBin() {
            p.executableURL = URL(fileURLWithPath: py)
            p.arguments = [
                "-m",
                "precursor.cli.interviewer_agent_cli",
                "--project",
                projectName,
            ]
        } else if let condaBin = resolveCondaBin() {
            p.executableURL = URL(fileURLWithPath: condaBin)
            p.arguments = [
                "run",
                "-n",
                condaEnvName,
                "python",
                "-m",
                "precursor.cli.interviewer_agent_cli",
                "--project",
                projectName,
            ]
        } else {
            // Fallback: use /usr/bin/env so "python3" resolves via PATH.
            p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            p.arguments = [
                "python3",
                "-m",
                "precursor.cli.interviewer_agent_cli",
                "--project",
                projectName,
            ]
        }

        // Fire-and-forget: CLI persists the next question to the DB.
        DispatchQueue.global(qos: .utility).async {
            do {
                try p.run()
                p.waitUntilExit()
                let code = p.terminationStatus
                if code != 0 {
                    let errData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
                    let outData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                    let err = String(data: errData, encoding: .utf8) ?? ""
                    let out = String(data: outData, encoding: .utf8) ?? ""
                    DispatchQueue.main.async {
                        let detail = (err.isEmpty ? out : err).trimmingCharacters(in: .whitespacesAndNewlines)
                        if !detail.isEmpty {
                            self.errorMessage = "Interviewer CLI failed (\(code)): \(detail)"
                        } else {
                            self.errorMessage = "Interviewer CLI failed (\(code))."
                        }
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to launch interviewer CLI: \(error.localizedDescription)"
                }
            }
        }
    }

    func accept(_ task: AgentTaskItem) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try self.db.updateTaskSection(id: task.id, to: .accepted)
                DispatchQueue.main.async {
                    // update local copy
                    if let idx = self.tasks.firstIndex(of: task) {
                        self.tasks[idx] = AgentTaskItem(
                            id: task.id,
                            projectName: task.projectName,
                            status: .accepted,
                            message: task.message,
                            createdAt: task.createdAt,
                            metadata: task.metadata
                        )
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                }
            }
        }
    }

    func reject(_ task: AgentTaskItem) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Move to Rejected; app hides rejected
                try self.db.updateTaskSectionRaw(id: task.id, to: "Rejected Agent Completed Tasks")
                DispatchQueue.main.async {
                    self.tasks.removeAll { $0.id == task.id }
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                }
            }
        }
    }

    func trashConversation(projectName: String) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try self.db.trashConversation(projectName: projectName)
                let unread = try self.db.unreadConversationCountsByProject()
                DispatchQueue.main.async {
                    self.conversationMessages.removeAll { $0.projectName == projectName }
                    self.unreadConversationByProject = unread
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                }
            }
        }
    }
}

// MARK: - UI

struct PrecursorAppView: View {
    @ObservedObject var state: AppState
    let initialProject: String?
    @State private var expanded: Set<Int64> = []
    @State private var showSettings: Bool = false
    @State private var showConversation: Bool = false
    @State private var conversationProject: String? = nil
    @State private var showMemory: Bool = false
    @State private var memoryProject: String? = nil

    var headerGradient: some View {
        LinearGradient(colors: [Color(nsColor: .controlAccentColor).opacity(0.35), .black.opacity(0.6)],
                       startPoint: .topLeading, endPoint: .bottomTrailing)
            .ignoresSafeArea()
    }

    var body: some View {
        ZStack {
            headerGradient
            VStack(alignment: .leading, spacing: 18) {
                header
                projectPickerBar
                content
            }
            .padding(24)
        }
        .frame(minWidth: 1000, minHeight: 680)
        .onAppear {
            if let proj = initialProject {
                state.selectedProject = proj
            }
            state.loadInitial()
        }
        .onReceive(Timer.publish(every: 30, on: .main, in: .common).autoconnect()) { _ in
            // Background poll for badge counts while not actively viewing a conversation.
            if !showConversation {
                state.refreshUnreadConversationCounts()
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsSheetView(isPresented: $showSettings)
                .frame(minWidth: 760, minHeight: 540)
        }
        .sheet(isPresented: $showConversation) {
            if let proj = conversationProject ?? state.selectedProject {
                ConversationSheetView(
                    state: state,
                    projectName: proj,
                    isPresented: $showConversation
                )
                .frame(minWidth: 920, minHeight: 680)
            } else {
                Text("No project selected").padding(24)
            }
        }
        .sheet(isPresented: $showMemory) {
            if let proj = memoryProject ?? state.selectedProject {
                MemorySheetView(
                    state: state,
                    projectName: proj,
                    isPresented: $showMemory
                )
                .frame(minWidth: 980, minHeight: 720)
            } else {
                Text("No project selected").padding(24)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Here's what I worked on for")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.secondary)
                Text(state.selectedProject ?? "—")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundStyle(.white)
                    .shadow(color: .black.opacity(0.25), radius: 8, x: 0, y: 2)
            }
            Spacer()
            ConversationBadgeButton(
                count: state.unreadConversationByProject[state.selectedProject ?? ""] ?? 0,
                label: "Conversation",
                compact: false,
                action: {
                    conversationProject = state.selectedProject
                    showConversation = true
                }
            )
            Button(action: {
                memoryProject = state.selectedProject
                showMemory = true
            }) {
                HStack(spacing: 8) {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 13, weight: .bold))
                    Text("Memory")
                        .font(.system(size: 13, weight: .semibold))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
                .cornerRadius(10)
                .shadow(color: .black.opacity(0.18), radius: 6, x: 0, y: 4)
            }
            .buttonStyle(.plain)
            Capsule()
                .fill(.ultraThinMaterial)
                .overlay(
                    HStack(spacing: 10) {
                        Image(systemName: "bolt.fill")
                            .symbolRenderingMode(.palette)
                            .foregroundStyle(.yellow, .white.opacity(0.7))
                        Text("Precursor Agent Updates")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(.primary)
                    }
                    .padding(.horizontal, 14)
                )
                .frame(height: 36)
                .shadow(color: .black.opacity(0.25), radius: 8, x: 0, y: 4)
            Button(action: { showSettings = true }) {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 14, weight: .bold))
                    .padding(10)
                    .background(.ultraThinMaterial)
                    .foregroundStyle(.primary)
                    .cornerRadius(10)
                    .shadow(color: .black.opacity(0.2), radius: 6, x: 0, y: 4)
            }
            .buttonStyle(.plain)
        }
    }

    private var projectPickerBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(state.projects, id: \.self) { project in
                    let selected = (project == state.selectedProject)
                    Button(action: { state.selectProject(project) }) {
                        let unread = state.unreadConversationByProject[project] ?? 0
                        ZStack(alignment: .topTrailing) {
                            HStack(spacing: 8) {
                                Image(systemName: selected ? "checkmark.seal.fill" : "seal")
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundStyle(selected ? .white : .secondary)
                                Text(project)
                                    .font(.system(size: 13, weight: .semibold))
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(
                                ZStack {
                                    if selected {
                                        LinearGradient(colors: [Color(nsColor: .controlAccentColor), Color(nsColor: .controlAccentColor).opacity(0.8)], startPoint: .leading, endPoint: .trailing)
                                    } else {
                                        Color.white.opacity(0.08)
                                    }
                                }
                            )
                            .foregroundStyle(selected ? .white : .primary)
                            .cornerRadius(10)
                            .shadow(color: selected ? Color(nsColor: .controlAccentColor).opacity(0.35) : .clear, radius: 8, x: 0, y: 4)

                            if unread > 0 {
                                Text("\(min(unread, 99))")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 3)
                                    .background(
                                        LinearGradient(
                                            colors: [Color(nsColor: .controlAccentColor), Color(nsColor: .controlAccentColor).opacity(0.75)],
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        )
                                    )
                                    .clipShape(Capsule())
                                    .overlay(Capsule().stroke(.white.opacity(0.18), lineWidth: 1))
                                    .offset(x: 6, y: -6)
                                    .shadow(color: Color(nsColor: .controlAccentColor).opacity(0.35), radius: 8, x: 0, y: 4)
                            }
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 2)
        }
    }

    private var content: some View {
        Group {
            if state.isLoading {
                ProgressView().progressViewStyle(.circular)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            } else if let error = state.errorMessage {
                Text(error).foregroundStyle(.red)
            } else if state.tasks.isEmpty {
                emptyState
            } else {
                ScrollView {
                    LazyVStack(spacing: 16) {
                        ForEach(groupTasks(state.tasks), id: \.0) { (title, items) in
                            if !items.isEmpty {
                                GroupHeader(title: title)
                                    .padding(.top, 6)
                                ForEach(items) { item in
                                    AgentTaskCard(
                                        item: item,
                                        isExpanded: expanded.contains(item.id),
                                        toggleExpanded: { toggleExpanded(for: item) },
                                        onView: { open(item) },
                                        onAccept: { state.accept(item) },
                                        onReject: { state.reject(item) }
                                    )
                                    .transition(.asymmetric(insertion: .opacity.combined(with: .scale(scale: 0.98)), removal: .move(edge: .trailing).combined(with: .opacity)))
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 48))
                .foregroundStyle(.green)
            Text("All caught up")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(.white)
            Text("No pending or accepted agent updates")
                .foregroundStyle(.secondary)
        }
        .padding(28)
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial)
        .cornerRadius(14)
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(.white.opacity(0.08), lineWidth: 1)
        )
    }

    private func toggleExpanded(for item: AgentTaskItem) {
        if expanded.contains(item.id) {
            expanded.remove(item.id)
        } else {
            expanded.insert(item.id)
        }
    }

    private func open(_ item: AgentTaskItem) {
        guard let uri = item.uri, !uri.isEmpty else { return }
        if let url = URL(string: uri), ["http", "https", "file"].contains(url.scheme?.lowercased() ?? "") {
            NSWorkspace.shared.open(url)
            return
        }
        let fileURL = URL(fileURLWithPath: uri)
        NSWorkspace.shared.open(fileURL)
    }
}

struct GroupHeader: View {
    let title: String
    var body: some View {
        HStack {
            Text(title)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 6)
    }
}

struct AgentTaskCard: View {
    let item: AgentTaskItem
    let isExpanded: Bool
    let toggleExpanded: () -> Void
    let onView: () -> Void
    let onAccept: () -> Void
    let onReject: () -> Void

    var statusPill: some View {
        let color: Color = (item.status == .pending) ? .yellow : .green
        let text: String = (item.status == .pending) ? "Pending Review" : "Accepted"
        return HStack(spacing: 6) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(text)
                .font(.system(size: 11, weight: .semibold))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.white.opacity(0.06))
        .cornerRadius(8)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                ZStack {
                    Circle()
                        .fill((item.status == .accepted ? Color.green : Color.yellow).opacity(0.18))
                        .frame(width: 36, height: 36)
                    Image(systemName: item.status == .accepted ? "checkmark" : "clock")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundStyle(item.status == .accepted ? .green : .yellow)
                }
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(item.shortDescription ?? item.taskTitle)
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundStyle(.primary)
                        Spacer()
                        statusPill
                    }
                    // Expanded details
                    if isExpanded {
                        if let steps = item.stepByStepSummary, !steps.isEmpty {
                            Text(steps)
                                .font(.system(size: 12))
                                .foregroundStyle(.secondary)
                        }
                        Text("(Agent Task: \(item.taskTitle))")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                    }
                    let hasDetails = (item.stepByStepSummary?.isEmpty == false) || (!item.taskTitle.isEmpty)
                    Button(action: toggleExpanded) {
                        HStack(spacing: 4) {
                            Text(isExpanded ? "Show less" : "Show more")
                            Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                                .font(.system(size: 11, weight: .semibold))
                        }
                        .font(.system(size: 12, weight: .semibold))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.primary)
                    .opacity(hasDetails ? 1 : 0)
                }
                Spacer()
            }
            actionRow
        }
        .padding(18)
        .background(.ultraThinMaterial)
        .cornerRadius(14)
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(.white.opacity(0.08), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.2), radius: 10, x: 0, y: 6)
    }

    private var actionRow: some View {
        HStack(spacing: 10) {
            if let uri = item.uri, !uri.isEmpty {
                Button(action: onView) {
                    Label("View", systemImage: "arrow.up.right.square")
                        .font(.system(size: 13, weight: .semibold))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(
                            LinearGradient(colors: [Color(nsColor: .controlAccentColor), Color(nsColor: .controlAccentColor).opacity(0.8)], startPoint: .leading, endPoint: .trailing)
                        )
                        .foregroundStyle(.white)
                        .cornerRadius(10)
                        .shadow(color: Color(nsColor: .controlAccentColor).opacity(0.4), radius: 8, x: 0, y: 4)
                }
                .buttonStyle(.plain)
                .help(uri)
            }

            if item.status == .pending {
                Button(action: onAccept) {
                    Label("Accept", systemImage: "checkmark")
                        .font(.system(size: 13, weight: .semibold))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(Color.green.opacity(0.12))
                        .foregroundStyle(.green)
                        .cornerRadius(10)
                }
                .buttonStyle(.plain)

                Button(role: .destructive, action: onReject) {
                    Label("Reject", systemImage: "xmark")
                        .font(.system(size: 13, weight: .semibold))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(Color.red.opacity(0.12))
                        .foregroundStyle(.red)
                        .cornerRadius(10)
                }
                .buttonStyle(.plain)
            }

            Spacer()
        }
    }
}

// MARK: - Conversation UI

struct ConversationBadgeButton: View {
    let count: Int
    let label: String?
    let compact: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack(alignment: .topTrailing) {
                HStack(spacing: compact ? 0 : 8) {
                    Image(systemName: "bubble.left.and.bubble.right.fill")
                        .font(.system(size: compact ? 12 : 13, weight: .bold))
                        .foregroundStyle(.primary)
                    if let label, !compact {
                        Text(label)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(.primary)
                    }
                }
                .padding(.horizontal, compact ? 10 : 14)
                .padding(.vertical, compact ? 8 : 10)
                .background(.ultraThinMaterial)
                .cornerRadius(10)
                .shadow(color: .black.opacity(0.18), radius: 6, x: 0, y: 4)

                if count > 0 {
                    Text("\(min(count, 99))")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(
                            LinearGradient(
                                colors: [Color(nsColor: .controlAccentColor), Color(nsColor: .controlAccentColor).opacity(0.75)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .clipShape(Capsule())
                        .overlay(Capsule().stroke(.white.opacity(0.18), lineWidth: 1))
                        .offset(x: 6, y: -6)
                        .shadow(color: Color(nsColor: .controlAccentColor).opacity(0.4), radius: 8, x: 0, y: 5)
                }
            }
        }
        .buttonStyle(.plain)
    }
}

struct ChatBubble: View {
    let role: ConversationRole
    let message: String
    let createdAt: Date

    var isUser: Bool { role == .user }

    var body: some View {
        HStack {
            if isUser { Spacer(minLength: 40) }

            VStack(alignment: .leading, spacing: 6) {
                let senderLabel: String = {
                    switch role {
                    case .agent: return "Precursor"
                    case .user: return "You"
                    case .system: return "System"
                    }
                }()

                HStack(spacing: 8) {
                    Text(senderLabel)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(humanReadableTimestamp(createdAt))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.head)
                }

                Text(message)
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(isUser ? .white : .primary)
                    .textSelection(.enabled)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(
                Group {
                    if isUser {
                        LinearGradient(
                            colors: [
                                Color(nsColor: .controlAccentColor).opacity(0.95),
                                Color(nsColor: .controlAccentColor).opacity(0.75),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    } else {
                        Color.white.opacity(0.10)
                    }
                }
            )
            .background(.ultraThinMaterial.opacity(isUser ? 0 : 1))
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(.white.opacity(isUser ? 0.10 : 0.08), lineWidth: 1)
            )
            .shadow(color: .black.opacity(isUser ? 0.25 : 0.18), radius: 10, x: 0, y: 6)

            if !isUser { Spacer(minLength: 40) }
        }
        .padding(.horizontal, 8)
    }
}

struct ConversationSheetView: View {
    @ObservedObject var state: AppState
    let projectName: String
    @Binding var isPresented: Bool

    @State private var draft: String = ""
    @State private var autoScrollToken: Int = 0
    @State private var showTrashConfirm: Bool = false

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(nsColor: .controlAccentColor).opacity(0.22), .black.opacity(0.55)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 14) {
                header
                messages
                composer
            }
            .padding(20)
        }
        .onAppear {
            state.loadConversation(projectName: projectName, markSeen: true)
        }
        .onReceive(Timer.publish(every: 5, on: .main, in: .common).autoconnect()) { _ in
            // Aggressive poll while conversation is open.
            state.loadConversation(projectName: projectName, markSeen: true)
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Conversation")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.white)
                Text(projectName)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button(role: .destructive, action: { showTrashConfirm = true }) {
                Label("Delete", systemImage: "trash")
                    .font(.system(size: 12, weight: .semibold))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.red.opacity(0.10))
                    .cornerRadius(10)
            }
            .buttonStyle(.plain)
            .help("Clear the conversation history for this project.")
            .alert("Clear conversation history?", isPresented: $showTrashConfirm) {
                Button("Cancel", role: .cancel) {}
                Button("Delete", role: .destructive) {
                    state.trashConversation(projectName: projectName)
                }
            } message: {
                Text("Are you sure you want to clear the conversation history for this project?")
            }
            Button(action: {
                state.loadConversation(projectName: projectName, markSeen: true)
            }) {
                Label("Refresh", systemImage: "arrow.clockwise")
                    .font(.system(size: 12, weight: .semibold))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.white.opacity(0.08))
                    .cornerRadius(10)
            }
            .buttonStyle(.plain)

            Button(action: { isPresented = false }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(14)
        .background(.ultraThinMaterial)
        .cornerRadius(14)
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.08), lineWidth: 1))
        .shadow(color: .black.opacity(0.18), radius: 10, x: 0, y: 6)
    }

    private var messages: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    if state.isConversationLoading && state.conversationMessages.isEmpty {
                        ProgressView().progressViewStyle(.circular)
                            .padding(.top, 24)
                    }
                    ForEach(state.conversationMessages.filter { $0.projectName == projectName }) { msg in
                        ChatBubble(role: msg.role, message: msg.message, createdAt: msg.createdAt)
                            .id(msg.id)
                    }
                    Color.clear.frame(height: 1).id("bottom-\(autoScrollToken)")
                }
                .padding(.vertical, 12)
            }
            .background(Color.white.opacity(0.04))
            .background(.ultraThinMaterial.opacity(0.75))
            .cornerRadius(14)
            .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.08), lineWidth: 1))
            .shadow(color: .black.opacity(0.16), radius: 10, x: 0, y: 6)
            .onChange(of: state.conversationMessages.count) { _ in
                autoScrollToken += 1
                withAnimation(.easeOut(duration: 0.25)) {
                    proxy.scrollTo("bottom-\(autoScrollToken)", anchor: .bottom)
                }
            }
            .onAppear {
                // Initial scroll to bottom after first load.
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                    autoScrollToken += 1
                    proxy.scrollTo("bottom-\(autoScrollToken)", anchor: .bottom)
                }
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                TextField("Write a message…", text: $draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 14))
                    .lineLimit(1...5)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .background(Color.white.opacity(0.08))
                    .cornerRadius(12)
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.10), lineWidth: 1))

                Button(action: {
                    let msg = draft
                    draft = ""
                    state.sendUserMessage(projectName: projectName, text: msg, triggerInterviewer: true)
                }) {
                    Image(systemName: "paperplane.fill")
                        .font(.system(size: 14, weight: .bold))
                        .padding(12)
                        .background(
                            LinearGradient(
                                colors: [Color(nsColor: .controlAccentColor), Color(nsColor: .controlAccentColor).opacity(0.75)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .foregroundStyle(.white)
                        .cornerRadius(12)
                        .shadow(color: Color(nsColor: .controlAccentColor).opacity(0.35), radius: 10, x: 0, y: 6)
                }
                .buttonStyle(.plain)
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .opacity(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.45 : 1)
            }

            Text("Precursor will search and observe in the background. Responses can take several minutes. Feel free to leave this conversation and return later.")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .padding(14)
        .background(.ultraThinMaterial)
        .cornerRadius(14)
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.08), lineWidth: 1))
        .shadow(color: .black.opacity(0.14), radius: 10, x: 0, y: 6)
    }
}

// MARK: - App Entrypoint

struct PrecursorAppMain: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var state = AppState()
    let initialProjectFromCLI: String?

    init() {
        initialProjectFromCLI = Self.parseCLIArgs()
    }

    private static func parseCLIArgs() -> String? {
        let args = CommandLine.arguments
        guard args.count > 1 else { return nil }
        var i = 1
        while i < args.count {
            let arg = args[i]
            if arg == "--project", i + 1 < args.count {
                return args[i + 1]
            }
            i += 1
        }
        return nil
    }

    var body: some Scene {
        WindowGroup {
            PrecursorAppView(state: state, initialProject: initialProjectFromCLI)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

// main.swift is the executable entrypoint for this Swift Package.
// We intentionally avoid the `@main` attribute to keep this file as the entrypoint.
PrecursorAppMain.main()

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            // If no visible windows, bring back the main SwiftUI window(s)
            for window in sender.windows {
                window.makeKeyAndOrderFront(self)
            }
        }
        return true
    }
}

// MARK: - Settings models and YAML I/O

struct ProjectConfig: Identifiable, Hashable {
    let id = UUID()
    var name: String
    var description: String
    var agentEnabled: Bool
}

struct UserConfig {
    var name: String
    var description: String
    var agentGoals: String
}

struct SystemSettingsConfig {
    // Python / environment
    var condaEnvName: String
    var pythonBin: String  // optional; if set, used directly (supports .venv)

    var valueWeight: Double
    var feasibilityWeight: Double
    var userPreferenceAlignmentWeight: Double
    var maxDeployedTasks: Int
    var deploymentThreshold: Double
    var safetyThreshold: Int
    // Transition sensitivities
    var departureTimeThresholdMinutes: Double
    var departureMinEntriesPreviousSegment: Int
    var arrivalTimeThresholdMinutes: Double
    var arrivalMinEntriesCurrentSegment: Int
    // Observation source cooldown (seconds)
    var observationCooldownSeconds: Double
}

enum ConfigPathKind {
    case projects
    case user
    case settings
}

enum ConfigIO {
    static func resolvePath(_ kind: ConfigPathKind) -> URL? {
        let env = ProcessInfo.processInfo.environment
        switch kind {
        case .projects:
            if let p = env["PRECURSOR_PROJECTS_FILE"], !p.isEmpty { return URL(fileURLWithPath: p) }
        case .user:
            if let p = env["PRECURSOR_USER_FILE"], !p.isEmpty { return URL(fileURLWithPath: p) }
        case .settings:
            if let p = env["PRECURSOR_SETTINGS_FILE"], !p.isEmpty { return URL(fileURLWithPath: p) }
        }
        // Fallback: search upwards for src/precursor/config/<file>.yaml from CWD
        let fileName: String
        switch kind {
        case .projects: fileName = "projects.yaml"
        case .user: fileName = "user.yaml"
        case .settings: fileName = "settings.yaml"
        }
        let fm = FileManager.default
        var dir = URL(fileURLWithPath: fm.currentDirectoryPath)
        for _ in 0..<8 {
            let candidate = dir.appendingPathComponent("src/precursor/config/\(fileName)")
            if fm.fileExists(atPath: candidate.path) {
                return candidate
            }
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path { break }
            dir = parent
        }
        return nil
    }

    private static func existingHeader(at url: URL) -> String? {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path),
              let text = try? String(contentsOf: url, encoding: .utf8)
        else { return nil }
        var headerLines: [String] = []
        for raw in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(raw)
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("#") || trimmed.isEmpty {
                headerLines.append(line)
            } else {
                break
            }
        }
        return headerLines.isEmpty ? nil : headerLines.joined(separator: "\n") + "\n"
    }

    // Minimal YAML parsing tailored to our files
    static func loadUser() throws -> UserConfig {
        guard let path = resolvePath(.user) else { throw NSError(domain: "Config", code: 1, userInfo: [NSLocalizedDescriptionKey: "user.yaml not found"]) }
        let text = try String(contentsOf: path, encoding: .utf8)
        var name = ""
        var description = ""
        var goals = ""
        var currentKey: String?
        var collectingBlock: [String] = []
        func flushBlock() {
            guard let k = currentKey else { return }
            let joined = collectingBlock.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if k == "description" { description = joined }
            if k == "agent_goals" { goals = joined }
            currentKey = nil
            collectingBlock = []
        }
        for raw in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(raw)
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("#") { continue }
            if line.hasPrefix("name:") {
                flushBlock()
                name = line.replacingOccurrences(of: "name:", with: "").trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                continue
            }
            if line.hasPrefix("description: |") {
                flushBlock()
                currentKey = "description"
                collectingBlock = []
                continue
            }
            if line.hasPrefix("agent_goals: |") {
                flushBlock()
                currentKey = "agent_goals"
                collectingBlock = []
                continue
            }
            if let _ = currentKey {
                if line.hasPrefix("  ") || line.isEmpty {
                    collectingBlock.append(line.hasPrefix("  ") ? String(line.dropFirst(2)) : "")
                } else {
                    flushBlock()
                }
            }
        }
        flushBlock()
        return UserConfig(name: name, description: description, agentGoals: goals)
    }

    static func saveUser(_ u: UserConfig) throws {
        guard let path = resolvePath(.user) else { throw NSError(domain: "Config", code: 2, userInfo: [NSLocalizedDescriptionKey: "user.yaml path not resolved"]) }
        let preservedHeader = existingHeader(at: path)
        let defaultHeader = """
# config/user.yaml
# ---------------------------------------------------------------------------
# User profile / preferences
# ---------------------------------------------------------------------------
# This file is meant for LLM-facing components that want to tailor behavior
# to *you* (priorities, personality, preferences, etc).
# ---------------------------------------------------------------------------

"""
        let header = preservedHeader ?? defaultHeader
        let body =
"""
\(header)name: \"\(u.name)\"
description: |
  \(u.description.replacingOccurrences(of: "\n", with: "\n  "))
agent_goals: |
  \(u.agentGoals.replacingOccurrences(of: "\n", with: "\n  "))
"""
        try body.write(to: path, atomically: true, encoding: .utf8)
    }

    static func loadSettings() throws -> SystemSettingsConfig {
        guard let path = resolvePath(.settings) else { throw NSError(domain: "Config", code: 3, userInfo: [NSLocalizedDescriptionKey: "settings.yaml not found"]) }
        let text = try String(contentsOf: path, encoding: .utf8)
        var map: [String: String] = [:]
        for raw in text.split(separator: "\n") {
            let line = String(raw)
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("#") { continue }
            let parts = line.split(separator: ":", maxSplits: 1).map(String.init)
            if parts.count == 2 {
                map[parts[0].trimmingCharacters(in: .whitespaces)] = parts[1].trimmingCharacters(in: .whitespaces)
            }
        }
        func d(_ k: String, _ def: Double) -> Double { Double(map[k] ?? "") ?? def }
        func i(_ k: String, _ def: Int) -> Int { Int(map[k] ?? "") ?? def }
        func s(_ k: String, _ def: String) -> String {
            let raw = (map[k] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if raw.isEmpty { return def }
            // strip surrounding quotes if present
            return raw.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        }
        return SystemSettingsConfig(
            condaEnvName: s("conda_env_name", "gum"),
            pythonBin: s("python_bin", ""),
            valueWeight: d("value_weight", 2.0),
            feasibilityWeight: d("feasibility_weight", 1.5),
            userPreferenceAlignmentWeight: d("user_preference_alignment_weight", 0.5),
            maxDeployedTasks: i("max_deployed_tasks", 3),
            deploymentThreshold: d("deployment_threshold", 0.9),
            safetyThreshold: i("safety_threshold", 7),
            departureTimeThresholdMinutes: d("departure_time_threshold_minutes", 3.0),
            departureMinEntriesPreviousSegment: i("departure_min_entries_previous_segment", 3),
            arrivalTimeThresholdMinutes: d("arrival_time_threshold_minutes", 15.0),
            arrivalMinEntriesCurrentSegment: i("arrival_min_entries_current_segment", 1),
            observationCooldownSeconds: d("observation_cooldown_seconds", 60.0)
        )
    }

    static func saveSettings(_ s: SystemSettingsConfig) throws {
        guard let path = resolvePath(.settings) else { throw NSError(domain: "Config", code: 4, userInfo: [NSLocalizedDescriptionKey: "settings.yaml path not resolved"]) }
        let preservedHeader = existingHeader(at: path)
        let defaultHeader = """
# config/settings.yaml
# ---------------------------------------------------------------------------
# Settings for the system as a whole.
# ---------------------------------------------------------------------------
# This file is meant for settings that are used to configure the system.
# In particular the value, feasibility, safety, and user_preference alignment
# decide which tasks are considered for deployment.  The deployment threshold
# ---------------------------------------------------------------------------

"""
        let header = preservedHeader ?? defaultHeader
        let body =
"""
\(header)# Python runtime (used by the UI to launch the interviewer CLI)
conda_env_name: \"\(s.condaEnvName.replacingOccurrences(of: "\"", with: ""))\"
python_bin: \"\(s.pythonBin.replacingOccurrences(of: "\"", with: ""))\"

value_weight: \(formatDouble(s.valueWeight))
feasibility_weight: \(formatDouble(s.feasibilityWeight))
user_preference_alignment_weight: \(formatDouble(s.userPreferenceAlignmentWeight))

max_deployed_tasks: \(s.maxDeployedTasks)
deployment_threshold: \(formatDouble(s.deploymentThreshold))

safety_threshold: \(s.safetyThreshold)

# Notification / transition sensitivities
# ---------------------------------------------------------------------------
departure_time_threshold_minutes: \(formatDouble(s.departureTimeThresholdMinutes))
departure_min_entries_previous_segment: \(s.departureMinEntriesPreviousSegment)

arrival_time_threshold_minutes: \(formatDouble(s.arrivalTimeThresholdMinutes))
arrival_min_entries_current_segment: \(s.arrivalMinEntriesCurrentSegment)

# Observation source cooldown
# ---------------------------------------------------------------------------
observation_cooldown_seconds: \(formatDouble(s.observationCooldownSeconds))
"""
        try body.write(to: path, atomically: true, encoding: .utf8)
    }

    static func loadProjects() throws -> [ProjectConfig] {
        guard let path = resolvePath(.projects) else { throw NSError(domain: "Config", code: 5, userInfo: [NSLocalizedDescriptionKey: "projects.yaml not found"]) }
        let text = try String(contentsOf: path, encoding: .utf8)
        var projects: [ProjectConfig] = []
        var current: ProjectConfig?
        for raw in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(raw)
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("#") { continue }
            if line.trimmingCharacters(in: .whitespaces) == "projects:" {
                continue
            }
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("- name:") {
                if let c = current { projects.append(c) }
                let name = line.components(separatedBy: ":").dropFirst().joined(separator: ":").trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                current = ProjectConfig(name: name.trimmingCharacters(in: CharacterSet(charactersIn: "\"")), description: "", agentEnabled: true)
                continue
            }
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("description:") {
                let val = line.components(separatedBy: ":").dropFirst().joined(separator: ":").trimmingCharacters(in: .whitespaces)
                let desc = val.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                current?.description = desc
                continue
            }
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("agent_enabled:") {
                let val = line.components(separatedBy: ":").dropFirst().joined(separator: ":").trimmingCharacters(in: .whitespaces)
                current?.agentEnabled = (val.lowercased().hasPrefix("t"))
                continue
            }
        }
        if let c = current { projects.append(c) }
        return projects
    }

    static func saveProjects(_ projects: [ProjectConfig]) throws {
        guard let path = resolvePath(.projects) else { throw NSError(domain: "Config", code: 6, userInfo: [NSLocalizedDescriptionKey: "projects.yaml path not resolved"]) }
        let preservedHeader = existingHeader(at: path)
        var lines: [String] = []
        if let header = preservedHeader {
            lines.append(contentsOf: header.split(separator: "\n").map(String.init))
        } else {
            lines.append("# config/projects.yaml")
            lines.append("# ---------------------------------------------------------------------------")
            lines.append("# Project Registry")
            lines.append("# ---------------------------------------------------------------------------")
            lines.append("")
        }
        lines.append("projects:")
        for p in projects {
            lines.append("  - name: \"\(p.name)\"")
            lines.append("    description: \"\(p.description.replacingOccurrences(of: "\"", with: "\\\""))\"")
            lines.append("    agent_enabled: \(p.agentEnabled ? "true" : "false")")
            lines.append("")
        }
        try lines.joined(separator: "\n").write(to: path, atomically: true, encoding: .utf8)
    }

    private static func formatDouble(_ v: Double) -> String {
        if v.rounded(.toNearestOrAwayFromZero) == v { return String(format: "%.0f", v) }
        return String(format: "%.3f", v)
    }
}

final class SettingsViewModel: ObservableObject {
    @Published var projects: [ProjectConfig] = []
    @Published var user = UserConfig(name: "", description: "", agentGoals: "")
    @Published var settings = SystemSettingsConfig(
        condaEnvName: "gum",
        pythonBin: "",
        valueWeight: 2.0, feasibilityWeight: 1.5, userPreferenceAlignmentWeight: 0.5,
        maxDeployedTasks: 3, deploymentThreshold: 0.9, safetyThreshold: 7,
        departureTimeThresholdMinutes: 3.0, departureMinEntriesPreviousSegment: 3,
        arrivalTimeThresholdMinutes: 15.0, arrivalMinEntriesCurrentSegment: 1,
        observationCooldownSeconds: 60.0
    )
    @Published var errorMessage: String? = nil
    @Published var savedBanner: String? = nil

    func loadAll() {
        do {
            projects = try ConfigIO.loadProjects()
            user = try ConfigIO.loadUser()
            settings = try ConfigIO.loadSettings()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func saveProjects() {
        do {
            try ConfigIO.saveProjects(projects)
            savedBanner = "Projects saved"
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    func saveUser() {
        do {
            try ConfigIO.saveUser(user)
            savedBanner = "User profile saved"
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    func saveSettings() {
        do {
            try ConfigIO.saveSettings(settings)
            savedBanner = "System settings saved"
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

// MARK: - Settings UI

struct SettingsSheetView: View {
    @Binding var isPresented: Bool
    @StateObject private var vm = SettingsViewModel()
    @State private var tab: Int = 0

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(nsColor: .controlAccentColor).opacity(0.25), .black.opacity(0.5)], startPoint: .topLeading, endPoint: .bottomTrailing)
                .ignoresSafeArea()
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Label("Settings", systemImage: "gearshape.fill")
                        .font(.system(size: 22, weight: .bold))
                        .foregroundStyle(.white)
                        .shadow(color: .black.opacity(0.25), radius: 8, x: 0, y: 2)
                    Spacer()
                    Button(action: { isPresented = false }) {
                        Image(systemName: "xmark.circle.fill").font(.system(size: 18, weight: .semibold))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                }
                .padding(.bottom, 4)

                Picker("", selection: $tab) {
                    Text("Projects").tag(0)
                    Text("User Profile").tag(1)
                    Text("System").tag(2)
                }
                .pickerStyle(.segmented)

                Group {
                    if tab == 0 { ProjectsSettingsView(vm: vm) }
                    if tab == 1 { UserSettingsView(vm: vm) }
                    if tab == 2 { SystemSettingsView(vm: vm) }
                }
                .background(.ultraThinMaterial)
                .cornerRadius(14)
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.08), lineWidth: 1))
                .shadow(color: .black.opacity(0.2), radius: 8, x: 0, y: 6)

                if let err = vm.errorMessage, !err.isEmpty {
                    Text(err).foregroundStyle(.red)
                } else if let banner = vm.savedBanner {
                    Text(banner).foregroundStyle(.green).transition(.opacity)
                }
            }
            .padding(20)
        }
        .onAppear { vm.loadAll() }
    }
}

struct ProjectsSettingsView: View {
    @ObservedObject var vm: SettingsViewModel

    var body: some View {
        VStack(alignment: .leading) {
            HStack {
                Button {
                    vm.projects.append(ProjectConfig(name: "New Project", description: "", agentEnabled: true))
                } label: {
                    Label("Add Project", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
                Spacer()
                Button {
                    vm.saveProjects()
                } label: {
                    Label("Save", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.bordered)
            }
            .padding()

            ScrollView {
                VStack(spacing: 12) {
                    ForEach(Array(vm.projects.enumerated()), id: \.element.id) { index, _ in
                        ProjectRowEditor(
                            project: $vm.projects[index],
                            onDelete: { vm.projects.remove(at: index) }
                        )
                    }
                }
                .padding(12)
            }
        }
    }
}

struct ProjectRowEditor: View {
    @Binding var project: ProjectConfig
    var onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Project")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                TextField("Name", text: $project.name)
                    .textFieldStyle(.roundedBorder)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Description")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                TextEditor(text: $project.description)
                    .frame(minHeight: 80)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.white.opacity(0.15), lineWidth: 1)
                    )
            }
            Toggle(isOn: $project.agentEnabled) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Background agents enabled")
                        .font(.system(size: 12, weight: .semibold))
                    Text("Allow autonomous background tasks for this project.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            }
            .toggleStyle(.switch)
            HStack {
                Spacer()
                Button(role: .destructive) {
                    onDelete()
                } label: {
                    Label("Remove", systemImage: "trash")
                }
            }
        }
        .padding(12)
        .background(Color.white.opacity(0.05))
        .cornerRadius(10)
    }
}

struct UserSettingsView: View {
    @ObservedObject var vm: SettingsViewModel
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Spacer()
                Button {
                    vm.saveUser()
                } label: {
                    Label("Save", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.bordered)
            }
            .padding(12)
            Form {
                TextField("Name", text: $vm.user.name)
                VStack(alignment: .leading) {
                    Text("Description")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)
                    TextEditor(text: $vm.user.description)
                        .frame(minHeight: 100)
                }
                VStack(alignment: .leading) {
                    Text("Agent Goals")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)
                    TextEditor(text: $vm.user.agentGoals)
                        .frame(minHeight: 100)
                }
            }
            .formStyle(.grouped)
            .padding(12)
        }
    }
}

struct SystemSettingsView: View {
    @ObservedObject var vm: SettingsViewModel
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Spacer()
                Button {
                    vm.saveSettings()
                } label: {
                    Label("Save", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.bordered)
            }
            .padding(12)

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    GroupBox("Python Runtime") {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text("Python Executable")
                                    .frame(width: 160, alignment: .leading)
                                TextField("/path/to/python", text: $vm.settings.pythonBin)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 420)
                                Spacer()
                            }
                            Text("Optional. If set, the app will run using this Python directly (great for `./.venv/bin/python`). Leave blank to use conda.")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)

                            HStack {
                                Text("Conda Env Name")
                                    .frame(width: 160, alignment: .leading)
                                TextField("gum", text: $vm.settings.condaEnvName)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 220)
                                Spacer()
                            }
                            Text("Used by the macOS app when launching the interviewer CLI (via `conda run -n <env>`).")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                    }
                    GroupBox("Weights (0.0–5.0)") {
                        HStack {
                            Text("Value").frame(width: 160, alignment: .leading)
                            Slider(value: $vm.settings.valueWeight, in: 0.0...5.0, step: 0.25)
                            Text(String(format: "%.2f", vm.settings.valueWeight)).frame(width: 60, alignment: .trailing)
                        }
                        HStack {
                            Text("Feasibility").frame(width: 160, alignment: .leading)
                            Slider(value: $vm.settings.feasibilityWeight, in: 0.0...5.0, step: 0.25)
                            Text(String(format: "%.2f", vm.settings.feasibilityWeight)).frame(width: 60, alignment: .trailing)
                        }
                        HStack {
                            Text("Preference Alignment").frame(width: 160, alignment: .leading)
                            Slider(value: $vm.settings.userPreferenceAlignmentWeight, in: 0.0...5.0, step: 0.25)
                            Text(String(format: "%.2f", vm.settings.userPreferenceAlignmentWeight)).frame(width: 60, alignment: .trailing)
                        }
                    }
                    GroupBox("Deployment") {
                        HStack {
                            Text("Max Deployed Tasks").frame(width: 160, alignment: .leading)
                            TextField("", value: $vm.settings.maxDeployedTasks, formatter: NumberFormatter.integer)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 80)
                        }
                        HStack {
                            Text("Deployment Threshold").frame(width: 160, alignment: .leading)
                            Slider(value: $vm.settings.deploymentThreshold, in: 0.0...1.0, step: 0.05)
                            Text(String(format: "%.2f", vm.settings.deploymentThreshold)).frame(width: 60, alignment: .trailing)
                        }
                        HStack {
                            Text("Safety Threshold").frame(width: 160, alignment: .leading)
                            Slider(value: Binding(
                                get: { Double(vm.settings.safetyThreshold) },
                                set: { vm.settings.safetyThreshold = Int($0.rounded()) }
                            ), in: 1...10, step: 1)
                            Text("\(vm.settings.safetyThreshold)").frame(width: 60, alignment: .trailing)
                        }
                    }
                    GroupBox("Notifications & Agent Sensitivity") {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Departure (when leaving a project)").font(.system(size: 12, weight: .semibold))
                            HStack {
                                Text("Min Entries in Previous Segment").frame(width: 220, alignment: .leading)
                                TextField("", value: $vm.settings.departureMinEntriesPreviousSegment, formatter: NumberFormatter.integer)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 80)
                                Spacer()
                            }
                            HStack {
                                Text("Time Threshold (minutes)").frame(width: 220, alignment: .leading)
                                Slider(value: $vm.settings.departureTimeThresholdMinutes, in: 0...120, step: 5)
                                Text(String(format: "%.0f", vm.settings.departureTimeThresholdMinutes)).frame(width: 60, alignment: .trailing)
                            }
                            Divider().padding(.vertical, 4)
                            Text("Arrival (when returning to a project)").font(.system(size: 12, weight: .semibold))
                            HStack {
                                Text("Min Entries in Current Segment").frame(width: 220, alignment: .leading)
                                TextField("", value: $vm.settings.arrivalMinEntriesCurrentSegment, formatter: NumberFormatter.integer)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 80)
                                Spacer()
                            }
                            HStack {
                                Text("Absence Threshold (minutes)").frame(width: 220, alignment: .leading)
                                Slider(value: $vm.settings.arrivalTimeThresholdMinutes, in: 0...240, step: 5)
                                Text(String(format: "%.0f", vm.settings.arrivalTimeThresholdMinutes)).frame(width: 60, alignment: .trailing)
                            }
                            Divider().padding(.vertical, 4)
                            Text("Observation Cooldown (Gum)").font(.system(size: 12, weight: .semibold))
                            HStack {
                                Text("Cooldown (seconds)").frame(width: 220, alignment: .leading)
                                Slider(value: $vm.settings.observationCooldownSeconds, in: 0...600, step: 10)
                                Text(String(format: "%.0f", vm.settings.observationCooldownSeconds)).frame(width: 60, alignment: .trailing)
                            }
                        }
                    }
                }
                .padding(12)
            }
        }
    }
}

private extension NumberFormatter {
    static var integer: NumberFormatter {
        let nf = NumberFormatter()
        nf.numberStyle = .none
        nf.maximumFractionDigits = 0
        return nf
    }
}

// MARK: - Timestamp formatting

private func ordinalSuffix(_ day: Int) -> String {
    // 11th/12th/13th are special cases.
    let mod100 = day % 100
    if mod100 >= 11 && mod100 <= 13 { return "th" }
    switch day % 10 {
    case 1: return "st"
    case 2: return "nd"
    case 3: return "rd"
    default: return "th"
    }
}

private func humanReadableTimestamp(_ date: Date) -> String {
    // Use per-call DateFormatter instances to avoid thread-safety issues from shared
    // globals (this code runs while background polling updates the UI).
    let weekdayFormatter = DateFormatter()
    weekdayFormatter.locale = Locale(identifier: "en_US_POSIX")
    weekdayFormatter.dateFormat = "EEEE"

    let monthFormatter = DateFormatter()
    monthFormatter.locale = Locale(identifier: "en_US_POSIX")
    monthFormatter.dateFormat = "MMMM"

    let timeFormatter = DateFormatter()
    timeFormatter.locale = Locale(identifier: "en_US_POSIX")
    timeFormatter.dateFormat = "h:mm a"

    let cal = Calendar(identifier: .gregorian)
    let weekday = weekdayFormatter.string(from: date)
    let month = monthFormatter.string(from: date)
    let day = cal.component(.day, from: date)
    let year = cal.component(.year, from: date)
    let time = timeFormatter.string(from: date)

    return "\(weekday) \(month) \(day)\(ordinalSuffix(day)), \(year) (\(time))"
}


