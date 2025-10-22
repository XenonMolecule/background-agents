// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SurveyApp",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "SurveyApp", targets: ["SurveyApp"])
    ],
    targets: [
        .executableTarget(
            name: "SurveyApp",
            path: "Sources/SurveyApp"
        )
    ]
)


