## SurveyApp (macOS, SwiftUI)

This is a minimal SwiftUI macOS app that collects a short survey and appends results to `dev/survey/survey_responses.csv`.

### Questions
- What project are you working on right now?
- What task are you working on right now?
- What background context could be helpful with this task?
- What background context could be helpful with this project?
- What background work would have been helpful for this task?
- What background context would have been helpful with this project?

### Build & Run

Requirements: Xcode 15+ or Swift 5.9+ on macOS 13+

Option A — Run from Xcode:
1. Open this folder in Xcode (`File` → `Open...` → select `dev/survey/swiftui-survey`).
2. Xcode will detect the Swift Package. Select the `SurveyApp` scheme.
3. Run (⌘R).

Option B — Build and run via command line:
```bash
cd dev/survey/swiftui-survey
swift build -c release
.build/release/SurveyApp
```

### Output
Responses are appended to `dev/survey/survey_responses.csv` with the following columns:
`timestamp, project_now, task_now, helpful_task_context_now, helpful_project_context_now, helpful_task_background_work_past, helpful_project_background_context_past`


