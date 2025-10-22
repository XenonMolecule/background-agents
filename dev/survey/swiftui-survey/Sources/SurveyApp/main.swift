import SwiftUI
import AppKit

@main
struct SurveyAppMain: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 700, minHeight: 600)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

final class SurveyModel: ObservableObject {
    @Published var dataDir: String = FileManager.default.homeDirectoryForCurrentUser
        .appending(path: "Documents")
        .path()

    // Page 1
    @Published var projectNow: String = ""
    @Published var taskNow: String = ""

    // Page 2 (context that could be helpful)
    @Published var helpfulTaskContextNow: String = ""
    @Published var helpfulProjectContextNow: String = ""

    // Page 3 (background work that would have been helpful)
    @Published var helpfulTaskBackgroundWorkPast: String = ""
    @Published var helpfulProjectBackgroundWorkPast: String = ""

    func submitAndExit() {
        let csvHeaders = [
            "timestamp",
            "project_now",
            "task_now",
            "helpful_task_context_now",
            "helpful_project_context_now",
            "helpful_task_background_work_past",
            "helpful_project_background_work_past"
        ]

        let rows: [String] = [
            ISO8601DateFormatter().string(from: Date()),
            projectNow.replacingOccurrences(of: "\n", with: " "),
            taskNow.replacingOccurrences(of: "\n", with: " "),
            helpfulTaskContextNow.replacingOccurrences(of: "\n", with: " "),
            helpfulProjectContextNow.replacingOccurrences(of: "\n", with: " "),
            helpfulTaskBackgroundWorkPast.replacingOccurrences(of: "\n", with: " "),
            helpfulProjectBackgroundWorkPast.replacingOccurrences(of: "\n", with: " ")
        ].map { $0.replacingOccurrences(of: "\"", with: "\"\"") }

        let csvLine = rows.map { "\"\($0)\"" }.joined(separator: ",") + "\n"

        // Prefer writing into repo's dev/survey directory (sibling of this package dir)
        // Compute from source path (#filePath) for stability across run contexts
        let sourceURL = URL(fileURLWithPath: #filePath)
        let swiftuiSurveyDir = sourceURL
            .deletingLastPathComponent() // .../Sources/SurveyApp
            .deletingLastPathComponent() // .../Sources
            .deletingLastPathComponent() // .../swiftui-survey
        let surveyDir = swiftuiSurveyDir
            .deletingLastPathComponent()  // .../survey

        let outDir: URL
        if FileManager.default.fileExists(atPath: surveyDir.path()) {
            outDir = surveyDir
        } else {
            // Fallback: relative to current working directory
            let baseDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            outDir = baseDir
                .appending(path: "dev")
                .appending(path: "survey")
        }
        let outFile = outDir.appending(path: "survey_responses.csv")

        do {
            try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: outFile.path()) {
                let header = csvHeaders.joined(separator: ",") + "\n"
                try header.data(using: .utf8)?.write(to: outFile)
            }
            if let data = csvLine.data(using: .utf8) {
                if let handle = try? FileHandle(forWritingTo: outFile) {
                    try handle.seekToEnd()
                    try handle.write(contentsOf: data)
                    try handle.close()
                } else {
                    try data.write(to: outFile)
                }
            }
        } catch {
            // Silent failure to ensure app exits even if write fails
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            NSApp.terminate(nil)
        }
    }
}

struct ContentView: View {
    @StateObject var model = SurveyModel()
    @State private var page: Int = 1

    var body: some View {
        ZStack {
            Color.black.opacity(0.5).ignoresSafeArea()
            VStack(alignment: .leading, spacing: 16) {
                if page == 1 {
                    PageOne(model: model)
                } else if page == 2 {
                    PageTwo(model: model)
                } else {
                    PageThree(model: model)
                }
                HStack {
                    if page > 1 {
                        Button("Back") { page -= 1 }
                    }
                    Spacer()
                    if page < 3 {
                        Button("Next") { page += 1 }
                            .keyboardShortcut(.defaultAction)
                    } else {
                        Button("Submit") { model.submitAndExit() }
                            .keyboardShortcut(.defaultAction)
                    }
                }
            }
            .padding(20)
            .background(.regularMaterial)
            .cornerRadius(12)
            .shadow(radius: 12)
        }
    }
}

struct PageOne: View {
    @ObservedObject var model: SurveyModel
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What project are you working on right now?")
            TextEditor(text: $model.projectNow)
                .frame(height: 100)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary))
            Text("What task are you working on right now?")
            TextEditor(text: $model.taskNow)
                .frame(height: 100)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary))
        }
    }
}

struct PageTwo: View {
    @ObservedObject var model: SurveyModel
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What background context could be helpful with this task?")
            TextEditor(text: $model.helpfulTaskContextNow)
                .frame(height: 100)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary))
            Text("What background context could be helpful with this project?")
            TextEditor(text: $model.helpfulProjectContextNow)
                .frame(height: 100)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary))
        }
    }
}

struct PageThree: View {
    @ObservedObject var model: SurveyModel
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What background work would have been helpful for this task?")
            TextEditor(text: $model.helpfulTaskBackgroundWorkPast)
                .frame(height: 100)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary))
            Text("What background work would have been helpful with this project?")
            TextEditor(text: $model.helpfulProjectBackgroundWorkPast)
                .frame(height: 100)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary))
        }
    }
}


