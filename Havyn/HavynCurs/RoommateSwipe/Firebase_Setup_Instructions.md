# Firebase Setup Instructions

## Current Status

The app currently has Firebase dependencies for:
- FirebaseCore
- FirebaseFirestore
- FirebaseAuth
- FirebaseStorage

However, it's missing the `FirebaseFirestoreSwift` package which provides Codable support for Firestore, making it easier to convert between Swift objects and Firestore documents.

## How to Add FirebaseFirestoreSwift

1. Open your project in Xcode
2. Click on **File** > **Add Packages...**
3. In the search field, paste: `https://github.com/firebase/firebase-ios-sdk`
4. Select the Firebase iOS SDK package
5. In the package options on the right, make sure these packages are selected:
   - FirebaseCore
   - FirebaseFirestore 
   - FirebaseFirestoreSwift (this is the one we need to add)
   - FirebaseAuth (if you're using authentication)
   - FirebaseStorage (if you're storing files)
6. Click **Add Package**
7. Select your app target when prompted
8. Let Xcode resolve dependencies and download the package

## Update FirebaseService.swift

Once the package is added, you can update the FirebaseService.swift file to use the Codable support:

```swift
import Foundation
import FirebaseCore
import FirebaseFirestore
import FirebaseFirestoreSwift  // Re-add this import

class FirebaseService {
    // ... existing code ...
    
    /// Save or update a user profile in Firestore with Codable
    func saveUserProfileWithCodable(_ profile: UserProfile, userId: String, completion: @escaping (Error?) -> Void) {
        do {
            // Use the Codable extension to convert directly to Firestore document
            try db.collection(userCollection).document(userId).setData(from: profile) { error in
                completion(error)
            }
        } catch {
            completion(error)
        }
    }
    
    /// Fetch a user profile from Firestore with Codable
    func fetchUserProfileWithCodable(userId: String, completion: @escaping (Result<UserProfile, Error>) -> Void) {
        db.collection(userCollection).document(userId).getDocument { document, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            do {
                if let document = document, document.exists {
                    // Use the Decodable extension to convert directly from Firestore document
                    let profile = try document.data(as: UserProfile.self)
                    completion(.success(profile))
                } else {
                    completion(.failure(NSError(domain: "FirebaseService", code: 404, userInfo: [NSLocalizedDescriptionKey: "User not found"])))
                }
            } catch {
                completion(.failure(error))
            }
        }
    }
}
```

## Making UserProfile Codable

For the Codable extensions to work, your UserProfile model needs to conform to Codable:

```swift
struct UserProfile: Identifiable, Codable {
    var id = UUID().uuidString
    var name: String
    var age: Int
    // ... rest of your properties ...
    
    enum CodingKeys: String, CodingKey {
        case id
        case name
        case age
        // ... define keys for all properties ...
    }
}
```

## Benefits of Using FirebaseFirestoreSwift

1. **Simpler Code**: Convert between Swift objects and Firestore documents with just one line of code
2. **Type Safety**: Avoid errors from manual type casting
3. **Automatic Handling**: Field name mapping and optional values are handled automatically
4. **Custom Encoding/Decoding**: You can customize how properties are stored in Firestore

## Temporary Solution

Until you add the FirebaseFirestoreSwift package, the current manual implementation will continue to work. 