import Cocoa
import PlaygroundSupport


import CreateML
import Foundation

// 1️⃣ Point to your TrainingData folder:
let dataURL = URL(fileURLWithPath: "/Users/parthdangi/Downloads/FocusSense/TrainingData")

// 2️⃣ Load it as labeled directories:
let trainingData = try MLDataTable(labeledDirectories: dataURL)

// 3️⃣ Train the sound classifier:
let classifier = try MLSoundClassifier(
  trainingData: trainingData,
  parameters: .init(
    validationData: nil,
    algorithm: .randomForest,
    featureExtractor: .mfcc,
    maxIterations: 20
  )
)

// 4️⃣ Add metadata:
let metadata = MLModelMetadata(
  author: "Parth Dangi",
  shortDescription: "Ambient context classifier for FocusSense",
  version: "1.0"
)

// 5️⃣ Export to your Xcode project’s Models folder:
let exportURL = URL(fileURLWithPath:
  "/Users/parthdangi/Downloads/FocusSense/FocusSense/Models/FocusSenseSoundClassifier.mlmodel"
)
try classifier.write(to: exportURL, metadata: metadata)

// ✅ Success!
print("Model exported to \(exportURL.path)")

