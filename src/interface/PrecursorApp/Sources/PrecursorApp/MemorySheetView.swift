import SwiftUI
import UniformTypeIdentifiers

private enum MemoryMode: String {
    case read = "Read"
    case edit = "Edit"
}

struct MemorySheetView: View {
    @ObservedObject var state: AppState
    let projectName: String
    @Binding var isPresented: Bool

    @State private var mode: MemoryMode = .read

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(nsColor: .controlAccentColor).opacity(0.18), .black.opacity(0.55)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 14) {
                header
                if let err = state.memoryErrorMessage, !err.isEmpty {
                    Text(err).foregroundStyle(.red)
                }

                if mode == .read {
                    MemoryReadOnlyView(
                        entries: state.scratchpadEntries
                    )
                } else {
                    MemoryEditorView(state: state, projectName: projectName)
                }
            }
            .padding(20)
        }
        .onAppear {
            state.loadScratchpad(projectName: projectName)
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Memory")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.white)
                Text(projectName)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            Spacer()

            Picker("", selection: $mode) {
                Text("Read").tag(MemoryMode.read)
                Text("Edit").tag(MemoryMode.edit)
            }
            .pickerStyle(.segmented)
            .frame(width: 180)

            Button(action: { state.loadScratchpad(projectName: projectName) }) {
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
}

// MARK: - Read-only

private struct MemoryReadOnlyView: View {
    let entries: [ScratchpadEntry]

    @State private var expanded: Set<String> = ["Ongoing Objectives"]
    private let alwaysShowSections: Set<String> = Set(ScratchpadSchema.agentTaskSections)

    private var sectionsToShow: [String] {
        // Always apply the hidden-section filter so phased-out sections never appear.
        return ScratchpadSchema.userVisibleSections.filter { !ScratchpadSchema.defaultHiddenSections.contains($0) }
    }

    private var entriesBySection: [String: [ScratchpadEntry]] {
        var grouped = Dictionary(grouping: entries, by: { $0.section })
        for (k, v) in grouped {
            grouped[k] = v.sorted(by: { $0.sortOrder < $1.sortOrder })
        }
        return grouped
    }

    var body: some View {
        ScrollView {
            MemoryReadOnlyContent(
                entries: entries,
                sectionsToShow: sectionsToShow,
                entriesBySection: entriesBySection,
                alwaysShowSections: alwaysShowSections,
                expanded: $expanded
            )
            .padding(2)
        }
        .background(Color.white.opacity(0.04))
        .background(.ultraThinMaterial.opacity(0.75))
        .cornerRadius(14)
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.08), lineWidth: 1))
        .shadow(color: .black.opacity(0.16), radius: 10, x: 0, y: 6)
    }
}

private struct MemoryReadOnlyContent: View {
    let entries: [ScratchpadEntry]
    let sectionsToShow: [String]
    let entriesBySection: [String: [ScratchpadEntry]]
    let alwaysShowSections: Set<String>
    @Binding var expanded: Set<String>

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if entries.isEmpty {
                Text("No memory entries yet.")
                    .foregroundStyle(.secondary)
                    .padding(.top, 12)
            }

            ForEach(sectionsToShow, id: \.self) { section in
                MemorySectionCardIfNeeded(
                    section: section,
                    entries: entriesBySection[section] ?? [],
                    alwaysShow: alwaysShowSections.contains(section),
                    expanded: $expanded
                )
            }
        }
    }
}

private struct MemorySectionCardIfNeeded: View {
    let section: String
    let entries: [ScratchpadEntry]
    let alwaysShow: Bool
    @Binding var expanded: Set<String>

    var body: some View {
        if entries.isEmpty && !alwaysShow {
            EmptyView()
        } else {
            MemorySectionCard(section: section, entries: entries, expanded: $expanded)
        }
    }
}

private struct MemorySectionCard: View {
    let section: String
    let entries: [ScratchpadEntry]
    @Binding var expanded: Set<String>

    var body: some View {
        DisclosureGroup(
            isExpanded: Binding(
                get: { expanded.contains(section) },
                set: { isOn in
                    if isOn { expanded.insert(section) } else { expanded.remove(section) }
                }
            )
        ) {
            VStack(alignment: .leading, spacing: 10) {
                if entries.isEmpty {
                    Text("None")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                        .padding(.vertical, 2)
                } else
                if section == "Project Resources" {
                    MemoryConfidenceLegendRow()
                    MemoryProjectResourcesBlock(entries: entries)
                } else {
                    MemoryConfidenceLegendRow()
                    ForEach(entries) { e in
                        MemoryEntryLine(entry: e)
                    }
                }
            }
            .padding(.top, 8)
        } label: {
            HStack {
                Text(section)
                    .font(.system(size: 13, weight: .semibold))
                Spacer()
                Text("\(entries.count)")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .background(Color.white.opacity(0.06))
        .background(.ultraThinMaterial.opacity(0.65))
        .cornerRadius(14)
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.08), lineWidth: 1))
    }
}

private struct MemoryConfidenceLegendRow: View {
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "gauge")
                .font(.system(size: 11, weight: .semibold))
            Text("Confidence")
                .font(.system(size: 11, weight: .semibold))
            Text("(0–10)")
                .font(.system(size: 11, weight: .medium))
            Spacer()
        }
        .foregroundStyle(.secondary)
        .padding(.bottom, 2)
        .help("The number beside each memory is its confidence (0 = low, 10 = high).")
    }
}

private struct MemoryProjectResourcesBlock: View {
    let entries: [ScratchpadEntry]

    private var bySubsection: [String: [ScratchpadEntry]] {
        var grouped = Dictionary(grouping: entries, by: { $0.subsection ?? "Other" })
        for (k, v) in grouped {
            grouped[k] = v.sorted(by: { $0.sortOrder < $1.sortOrder })
        }
        return grouped
    }

    var body: some View {
        ForEach(ScratchpadSchema.resourceSubsections, id: \.self) { sub in
            if let rows = bySubsection[sub], !rows.isEmpty {
                Text(sub)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.top, 4)
                ForEach(rows) { e in
                    MemoryEntryLine(entry: e)
                }
            }
        }
    }
}

private struct MemoryEntryLine: View {
    let entry: ScratchpadEntry

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            ConfidencePill(confidence: entry.confidence)
            Text(entry.message)
                .font(.system(size: 13))
                .foregroundStyle(.primary)
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
        .padding(.vertical, 2)
    }
}

private struct ConfidencePill: View {
    let confidence: Int

    private var color: Color {
        switch confidence {
        case 8...10: return .green
        case 5...7: return .yellow
        default: return .secondary
        }
    }

    var body: some View {
        Text("\(confidence)")
            .font(.system(size: 11, weight: .bold))
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Color.white.opacity(0.06))
            .clipShape(Capsule())
            .overlay(Capsule().stroke(.white.opacity(0.10), lineWidth: 1))
            .frame(width: 42, alignment: .center)
            .help("Confidence: \(confidence)/10")
    }
}

// MARK: - Editor (Power users)

private struct MemoryEditorView: View {
    @ObservedObject var state: AppState
    let projectName: String

    @State private var selectedSection: String = ScratchpadSchema.userVisibleSections.first ?? "Notes"
    @State private var selectedSubsection: String = "Other"

    @State private var draftNewText: String = ""
    @State private var draftNewConfidence: Double = 7
    @State private var localEntries: [ScratchpadEntry] = []
    @State private var draggingId: Int64? = nil

    private var sectionPickerOptions: [String] { ScratchpadSchema.editorSections }

    private var entriesForSelection: [ScratchpadEntry] {
        state.scratchpadEntries
            .filter { e in
                guard e.section == selectedSection else { return false }
                if selectedSection == "Project Resources" {
                    return (e.subsection ?? "Other") == selectedSubsection
                }
                return true
            }
            .sorted(by: { $0.sortOrder < $1.sortOrder })
    }

    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                Picker("Section", selection: $selectedSection) {
                    ForEach(sectionPickerOptions, id: \.self) { s in
                        Text(s).tag(s)
                    }
                }
                .frame(width: 320)

                if selectedSection == "Project Resources" {
                    Picker("Subsection", selection: $selectedSubsection) {
                        ForEach(ScratchpadSchema.resourceSubsections, id: \.self) { s in
                            Text(s).tag(s)
                        }
                    }
                    .frame(width: 220)
                }

                Spacer()
            }

            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Entries (\(localEntries.count))")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 4)

                    VStack(spacing: 10) {
                        ForEach(localEntries) { e in
                            ScratchpadEntryEditorRow(
                                entry: e,
                                onSave: { newMsg, newConf in
                                    state.updateScratchpadEntry(projectName: projectName, id: e.id, message: newMsg, confidence: newConf)
                                },
                                onDelete: {
                                    state.deleteScratchpadEntry(projectName: projectName, id: e.id)
                                }
                            )
                            .padding(12)
                            .background(Color.white.opacity(0.05))
                            .cornerRadius(12)
                            .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.08), lineWidth: 1))
                            .onDrag {
                                draggingId = e.id
                                return NSItemProvider(object: String(e.id) as NSString)
                            }
                            .onDrop(
                                of: [UTType.plainText],
                                delegate: ScratchpadEntryDropDelegate(
                                    target: e,
                                    items: $localEntries,
                                    draggingId: $draggingId,
                                    onReorderCommitted: commitReorder
                                )
                            )
                        }
                    }

                    Divider().padding(.vertical, 8)

                    Text("Add new")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 4)

                    VStack(alignment: .leading, spacing: 10) {
                        TextField("New memory entry…", text: $draftNewText, axis: .vertical)
                            .lineLimit(2...8)
                            .textFieldStyle(.roundedBorder)

                        HStack {
                            Text("Confidence")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.secondary)
                            Slider(value: $draftNewConfidence, in: 0...10, step: 1)
                            Text("\(Int(draftNewConfidence))")
                                .font(.system(size: 12, weight: .semibold))
                                .frame(width: 28, alignment: .trailing)
                        }

                        Button {
                            let sub = (selectedSection == "Project Resources") ? selectedSubsection : nil
                            state.addScratchpadEntry(
                                projectName: projectName,
                                section: selectedSection,
                                subsection: sub,
                                message: draftNewText,
                                confidence: Int(draftNewConfidence)
                            )
                            draftNewText = ""
                            draftNewConfidence = 7
                        } label: {
                            Label("Add", systemImage: "plus.circle.fill")
                        }
                        .disabled(draftNewText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    .padding(12)
                    .background(Color.white.opacity(0.05))
                    .cornerRadius(12)
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.08), lineWidth: 1))
                }
                .padding(12)
            }
        }
        .onAppear {
            state.loadScratchpad(projectName: projectName)
            syncLocalEntries()
        }
        .onChange(of: selectedSection) { _ in syncLocalEntries() }
        .onChange(of: selectedSubsection) { _ in syncLocalEntries() }
        .onChange(of: state.scratchpadEntries) { _ in syncLocalEntries() }
        .background(Color.white.opacity(0.04))
        .background(.ultraThinMaterial.opacity(0.75))
        .cornerRadius(14)
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.08), lineWidth: 1))
        .shadow(color: .black.opacity(0.16), radius: 10, x: 0, y: 6)
    }

    private func syncLocalEntries() {
        localEntries = entriesForSelection
    }

    private func commitReorder() {
        let ids = localEntries.map { $0.id }
        let sub = (selectedSection == "Project Resources") ? selectedSubsection : nil
        state.reorderScratchpadGroup(projectName: projectName, section: selectedSection, subsection: sub, orderedIds: ids)
    }
}

private struct ScratchpadEntryDropDelegate: DropDelegate {
    let target: ScratchpadEntry
    @Binding var items: [ScratchpadEntry]
    @Binding var draggingId: Int64?
    let onReorderCommitted: () -> Void

    func dropEntered(info: DropInfo) {
        guard let draggingId,
              draggingId != target.id,
              let fromIndex = items.firstIndex(where: { $0.id == draggingId }),
              let toIndex = items.firstIndex(where: { $0.id == target.id })
        else { return }

        if items[toIndex].id != draggingId {
            withAnimation(.easeInOut(duration: 0.12)) {
                items.move(fromOffsets: IndexSet(integer: fromIndex), toOffset: (toIndex > fromIndex) ? (toIndex + 1) : toIndex)
            }
        }
    }

    func performDrop(info: DropInfo) -> Bool {
        draggingId = nil
        onReorderCommitted()
        return true
    }
}

private struct ScratchpadEntryEditorRow: View {
    let entry: ScratchpadEntry
    let onSave: (_ message: String, _ confidence: Int) -> Void
    let onDelete: () -> Void

    @State private var draftMessage: String
    @State private var draftConfidence: Double
    @State private var showDeleteConfirm: Bool = false

    init(entry: ScratchpadEntry, onSave: @escaping (String, Int) -> Void, onDelete: @escaping () -> Void) {
        self.entry = entry
        self.onSave = onSave
        self.onDelete = onDelete
        _draftMessage = State(initialValue: entry.message)
        _draftConfidence = State(initialValue: Double(entry.confidence))
    }

    private var isDirty: Bool {
        draftMessage.trimmingCharacters(in: .whitespacesAndNewlines) != entry.message.trimmingCharacters(in: .whitespacesAndNewlines)
            || Int(draftConfidence) != entry.confidence
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            TextField("Entry", text: $draftMessage, axis: .vertical)
                .lineLimit(1...6)

            HStack(spacing: 10) {
                Text("Confidence")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                Slider(value: $draftConfidence, in: 0...10, step: 1)
                Text("\(Int(draftConfidence))")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 28, alignment: .trailing)

                Spacer()

                Button {
                    onSave(draftMessage, Int(draftConfidence))
                } label: {
                    Label("Save", systemImage: "checkmark.circle.fill")
                }
                .disabled(!isDirty || draftMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                Button(role: .destructive) {
                    showDeleteConfirm = true
                } label: {
                    Label("Delete", systemImage: "trash")
                }
                .confirmationDialog("Delete this entry?", isPresented: $showDeleteConfirm) {
                    Button("Delete", role: .destructive) { onDelete() }
                    Button("Cancel", role: .cancel) {}
                }
            }
        }
        .padding(.vertical, 6)
        .onChange(of: entry.message) { newVal in
            if !isDirty {
                draftMessage = newVal
            }
        }
        .onChange(of: entry.confidence) { newVal in
            if !isDirty {
                draftConfidence = Double(newVal)
            }
        }
    }
}


