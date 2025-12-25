import Foundation

// MARK: - Scratchpad models (Project Memory)

struct ScratchpadEntry: Identifiable, Equatable, Hashable {
    let id: Int64
    let projectName: String
    let section: String
    let subsection: String?
    var message: String
    var confidence: Int
    var sortOrder: Int
    var lastEditedBy: String
    let createdAt: Date
}

enum ScratchpadSchema {
    // Keep in sync with `src/precursor/scratchpad/schema.py`
    static let projectSections: [String] = [
        "Ongoing Objectives",
        "Completed Objectives",
        "Suggestions",
        "Notes",
        "Project Resources",
        "Next Steps",
        "Agent Completed Tasks (Pending Review)",
        "Accepted Agent Completed Tasks",
        "Rejected Agent Completed Tasks",
    ]

    // Sections we are phasing out or hiding by default in the UI.
    // To re-enable a section, remove it from this set.
    static let defaultHiddenSections: Set<String> = [
        "Next Steps",
    ]

    // These are valuable to show in Memory even for general users.
    // We show them by default and we show the section card even if empty.
    static let agentTaskSections: [String] = [
        "Agent Completed Tasks (Pending Review)",
        "Accepted Agent Completed Tasks",
        "Rejected Agent Completed Tasks",
    ]

    static let resourceSubsections: [String] = [
        "Files",
        "Repos",
        "Folders",
        "Core Collaborators",
        "Other",
    ]

    // What the general user sees in Memory by default.
    static let userVisibleSections: [String] = {
        let base: [String] = [
            "Ongoing Objectives",
            "Completed Objectives",
            "Suggestions",
            "Notes",
            "Project Resources",
        ]
        return (base + agentTaskSections).filter { !defaultHiddenSections.contains($0) }
    }()

    // What the power-user editor allows editing by default (keeps workflow sections out).
    static let editorSections: [String] = {
        let base: [String] = [
            "Ongoing Objectives",
            "Completed Objectives",
            "Suggestions",
            "Notes",
            "Project Resources",
        ]
        return base.filter { !defaultHiddenSections.contains($0) }
    }()
}


