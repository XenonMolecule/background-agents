import Foundation

extension AppState {
    func loadScratchpad(projectName: String) {
        isMemoryLoading = true
        memoryErrorMessage = nil
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let items = try self.db.listScratchpadEntries(projectName: projectName)
                DispatchQueue.main.async {
                    self.scratchpadEntries = items
                    self.isMemoryLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.memoryErrorMessage = error.localizedDescription
                    self.isMemoryLoading = false
                }
            }
        }
    }

    func addScratchpadEntry(projectName: String, section: String, subsection: String?, message: String, confidence: Int) {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        isMemoryLoading = true
        memoryErrorMessage = nil
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                _ = try self.db.addScratchpadEntry(
                    projectName: projectName,
                    section: section,
                    subsection: subsection,
                    message: trimmed,
                    confidence: confidence
                )
                let items = try self.db.listScratchpadEntries(projectName: projectName)
                DispatchQueue.main.async {
                    self.scratchpadEntries = items
                    self.isMemoryLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.memoryErrorMessage = error.localizedDescription
                    self.isMemoryLoading = false
                }
            }
        }
    }

    func updateScratchpadEntry(projectName: String, id: Int64, message: String, confidence: Int) {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try self.db.updateScratchpadEntry(id: id, message: trimmed, confidence: confidence)
                let items = try self.db.listScratchpadEntries(projectName: projectName)
                DispatchQueue.main.async {
                    self.scratchpadEntries = items
                }
            } catch {
                DispatchQueue.main.async {
                    self.memoryErrorMessage = error.localizedDescription
                }
            }
        }
    }

    func deleteScratchpadEntry(projectName: String, id: Int64) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try self.db.deleteScratchpadEntry(id: id)
                let items = try self.db.listScratchpadEntries(projectName: projectName)
                DispatchQueue.main.async {
                    self.scratchpadEntries = items
                }
            } catch {
                DispatchQueue.main.async {
                    self.memoryErrorMessage = error.localizedDescription
                }
            }
        }
    }

    func reorderScratchpadGroup(
        projectName: String,
        section: String,
        subsection: String?,
        orderedIds: [Int64]
    ) {
        // Optimistically update local ordering so UI stays snappy.
        var idxById: [Int64: Int] = [:]
        for (idx, id) in orderedIds.enumerated() {
            idxById[id] = idx
        }
        DispatchQueue.main.async {
            self.scratchpadEntries = self.scratchpadEntries.map { e in
                guard e.projectName == projectName,
                      e.section == section,
                      (e.subsection ?? "") == (subsection ?? ""),
                      let idx = idxById[e.id]
                else { return e }
                var copy = e
                copy.sortOrder = idx
                return copy
            }
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try self.db.reorderScratchpadEntries(
                    projectName: projectName,
                    section: section,
                    subsection: subsection,
                    orderedIds: orderedIds
                )
            } catch {
                DispatchQueue.main.async {
                    self.memoryErrorMessage = error.localizedDescription
                }
            }
        }
    }
}


