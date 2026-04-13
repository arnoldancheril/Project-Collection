//
//  FirebaseService.swift
//  RoommateSwipe
//

import Foundation
import FirebaseCore
import FirebaseFirestore

class FirebaseService {
    static let shared = FirebaseService()
    
    private let db = Firestore.firestore()
    
    // Collections for different user types
    private let userCollection = "users"
    private let lookingForRoomCollection = "looking_for_room"
    private let haveRoomCollection = "have_room"
    private let apartmentListingCollection = "apartment_listings"
    private let sampleProfilesCollection = "sample_profiles"
    
    private init() {
        // Private initializer for singleton
    }
    
    // MARK: - User Profile Operations
    
    /// Save or update a user profile in Firestore
    func saveUserProfile(_ profile: UserProfile, userId: String, completion: @escaping (Error?) -> Void) {
        // Create a dictionary representation of the profile that can be stored in Firestore
        let userData: [String: Any] = [
            "name": profile.name,
            "age": profile.age,
            "city": profile.city,
            "budget": profile.budget,
            "interests": profile.interests,
            "bio": profile.bio,
            "email": profile.email,
            "phone": profile.phone,
            "hasPlace": profile.hasPlace,
            "moveInDate": profile.moveInDate,
            "smoking": profile.smoking,
            "pets": profile.pets,
            "wakeUpTime": profile.wakeUpTime,
            "sleepTime": profile.sleepTime,
            "cleanliness": profile.cleanliness,
            "socialLevel": profile.socialLevel,
            "imageName": profile.imageName,
            "profileImageName": profile.profileImageName ?? "",
            "propertyImageName": profile.propertyImageName ?? "",
            "lastUpdated": FieldValue.serverTimestamp()
        ]
        
        // Set the document with the user ID
        db.collection(userCollection).document(userId).setData(userData) { error in
            completion(error)
        }
    }
    
    /// Fetch a user profile from Firestore
    func fetchUserProfile(userId: String, completion: @escaping (Result<UserProfile, Error>) -> Void) {
        db.collection(userCollection).document(userId).getDocument { document, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            if let document = document, document.exists, let data = document.data() {
                // Create a UserProfile from the document data
                let profile = UserProfile(
                    name: data["name"] as? String ?? "",
                    age: data["age"] as? Int ?? 0,
                    city: data["city"] as? String ?? "",
                    budget: data["budget"] as? String ?? "",
                    interests: data["interests"] as? String ?? "",
                    bio: data["bio"] as? String ?? "",
                    email: data["email"] as? String ?? "",
                    phone: data["phone"] as? String ?? "",
                    hasPlace: data["hasPlace"] as? Bool ?? false,
                    moveInDate: data["moveInDate"] as? String ?? "",
                    smoking: data["smoking"] as? Bool ?? false,
                    pets: data["pets"] as? Bool ?? false,
                    wakeUpTime: data["wakeUpTime"] as? String ?? "",
                    sleepTime: data["sleepTime"] as? String ?? "",
                    cleanliness: data["cleanliness"] as? Int ?? 3,
                    socialLevel: data["socialLevel"] as? Int ?? 3,
                    imageName: data["imageName"] as? String ?? "",
                    profileImageName: data["profileImageName"] as? String,
                    propertyImageName: data["propertyImageName"] as? String
                )
                
                completion(.success(profile))
            } else {
                // Document doesn't exist
                completion(.failure(NSError(domain: "FirebaseService", code: 404, userInfo: [NSLocalizedDescriptionKey: "User not found"])))
            }
        }
    }
    
    /// Delete a user profile from Firestore
    func deleteUserProfile(userId: String, completion: @escaping (Error?) -> Void) {
        db.collection(userCollection).document(userId).delete { error in
            completion(error)
        }
    }
    
    // MARK: - User Registration
    
    /// Save registration data for a user looking for a room
    func saveLookingForRoomUser(_ data: UserRegistrationData, userId: String = UUID().uuidString, completion: @escaping (String, Error?) -> Void) {
        // Convert dates to strings for Firestore
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        
        let timeFormatter = DateFormatter()
        timeFormatter.timeStyle = .short
        
        // Create dictionary from registration data
        let userData: [String: Any] = [
            "userId": userId,
            "firstName": data.firstName,
            "lastName": data.lastName,
            "email": data.email,
            "dateOfBirth": dateFormatter.string(from: data.dateOfBirth),
            "gender": data.gender,
            "occupation": data.occupation,
            "bio": data.bio,
            "cleanliness": data.cleanliness,
            "noise": data.noise,
            "socialLevel": data.socialLevel,
            "sleepSchedule": data.sleepSchedule,
            "drinking": data.drinking,
            "smoking": data.smoking,
            "pets": data.pets,
            "preferredGender": data.preferredGender,
            "ageRangeMin": data.ageRange.lowerBound,
            "ageRangeMax": data.ageRange.upperBound,
            "rentRangeMin": data.rentRange.lowerBound,
            "rentRangeMax": data.rentRange.upperBound,
            "moveInDate": dateFormatter.string(from: data.moveInDate),
            "wakeUpTime": timeFormatter.string(from: data.wakeUpTime),
            "bedTime": timeFormatter.string(from: data.bedTime),
            "workSchedule": data.workSchedule,
            "preferredNeighborhoods": data.preferredNeighborhoods,
            "desiredAmenities": data.desiredAmenities,
            "roomPreference": data.roomPreference,
            "hasProfilePhoto": data.profilePhoto != nil,
            "registrationDate": FieldValue.serverTimestamp(),
            "userType": "lookingForRoom"
        ]
        
        // Save to the lookingForRoom collection
        db.collection(lookingForRoomCollection).document(userId).setData(userData) { error in
            completion(userId, error)
        }
    }
    
    /// Save registration data for a user who has a room
    func saveHaveRoomUser(_ data: HaveRoomRegistrationData, userId: String = UUID().uuidString, completion: @escaping (String, Error?) -> Void) {
        // Convert dates to strings
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        
        // Create dictionary from registration data
        let userData: [String: Any] = [
            "userId": userId,
            "fullName": data.fullName,
            "email": data.email,
            "phoneNumber": data.phoneNumber,
            "dateOfBirth": dateFormatter.string(from: data.dateOfBirth),
            "city": data.city,
            "neighborhood": data.neighborhood,
            "moveInDate": dateFormatter.string(from: data.moveInDate),
            "rentPrice": data.rentPrice,
            "housingType": data.housingType,
            "isFurnished": data.isFurnished,
            "genderPreference": data.genderPreference,
            "smokingPreference": data.smokingPreference,
            "petPreference": data.petPreference,
            "lifestylePreference": data.lifestylePreference,
            "cleanlinessLevel": data.cleanlinessLevel,
            "bio": data.bio,
            "interests": data.interests,
            "hasProfilePhoto": data.profilePhoto != nil,
            "hasRoomPhotos": !data.roomPhotos.isEmpty,
            "roomPhotoCount": data.roomPhotos.count,
            "registrationDate": FieldValue.serverTimestamp(),
            "userType": "haveRoom"
        ]
        
        // Save to the haveRoom collection
        db.collection(haveRoomCollection).document(userId).setData(userData) { error in
            completion(userId, error)
        }
    }
    
    /// Save registration data for an apartment listing
    func saveApartmentListing(_ data: ApartmentListingRegistrationData, listingId: String = UUID().uuidString, completion: @escaping (String, Error?) -> Void) {
        // Convert dates to strings
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        
        // Create dictionary from registration data
        let listingData: [String: Any] = [
            "listingId": listingId,
            "companyName": data.companyName,
            "contactPersonName": data.contactPersonName,
            "email": data.email,
            "contactPhoneNumber": data.contactPhoneNumber,
            "city": data.city,
            "neighborhood": data.neighborhood,
            "apartmentName": data.apartmentName,
            "address": data.address,
            "numberOfAvailableRooms": data.numberOfAvailableRooms,
            "availabilityDate": dateFormatter.string(from: data.availabilityDate),
            "minPrice": data.minPrice,
            "maxPrice": data.maxPrice,
            "housingType": data.housingType,
            "amenities": data.amenities,
            "isPetFriendly": data.isPetFriendly,
            "isSmokingAllowed": data.isSmokingAllowed,
            "propertyDescription": data.propertyDescription,
            "hasPropertyPhotos": !data.propertyPhotos.isEmpty,
            "propertyPhotoCount": data.propertyPhotos.count,
            "listingDate": FieldValue.serverTimestamp(),
            "userType": "apartmentListing"
        ]
        
        // Save to the apartmentListings collection
        db.collection(apartmentListingCollection).document(listingId).setData(listingData) { error in
            completion(listingId, error)
        }
    }
    
    // MARK: - Sample Profiles
    
    /// Upload sample profiles to Firestore
    func uploadSampleProfiles(profiles: [Profile], completion: @escaping (Error?) -> Void) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        
        let group = DispatchGroup()
        var lastError: Error?
        
        for profile in profiles {
            group.enter()
            
            // Create a dictionary representation of the profile that can be stored in Firestore
            var profileData: [String: Any] = [
                "id": profile.id.uuidString,
                "name": profile.name,
                "age": profile.age,
                "gender": profile.gender,
                "city": profile.city,
                "bio": profile.bio,
                "imageName": profile.imageName,
                "propertyImageName": profile.propertyImageName ?? "",
                "hasRoom": profile.hasRoom,
                "needsRoom": profile.needsRoom,
                "moveInDate": dateFormatter.string(from: profile.moveInDate),
                "preferredNeighborhoods": profile.preferredNeighborhoods,
                "budgetRangeMin": profile.budgetRange.lowerBound,
                "budgetRangeMax": profile.budgetRange.upperBound,
                "coordinate": [
                    "latitude": profile.coordinate.latitude,
                    "longitude": profile.coordinate.longitude
                ],
                "numberOfRooms": profile.numberOfRooms,
                "numberOfBathrooms": profile.numberOfBathrooms,
                "amenities": profile.amenities,
                "rent": profile.rent,
                "address": profile.address,
                "cleanliness": profile.cleanliness,
                "partying": profile.partying,
                "smoking": profile.smoking,
                "pets": profile.pets,
                "wakeUpTime": profile.wakeUpTime,
                "sleepTime": profile.sleepTime,
                "habits": profile.habits,
                "lookingFor": profile.lookingFor,
                "verificationStatus": profile.verificationStatus,
                "isBlocked": profile.isBlocked,
                "profileType": profile.profileType.rawValue,
                "uploadedAt": FieldValue.serverTimestamp()
            ]
            
            // Add pet types if available
            if let petTypes = profile.petTypes {
                profileData["petTypes"] = petTypes
            }
            
            // Generate random contact information
            profileData["email"] = "\(profile.name.lowercased().replacingOccurrences(of: " ", with: ""))@example.com"
            profileData["phone"] = generateRandomPhoneNumber()
            
            // Save to the sampleProfiles collection
            db.collection(sampleProfilesCollection).document(profile.id.uuidString).setData(profileData) { error in
                if let error = error {
                    print("Error uploading profile for \(profile.name): \(error.localizedDescription)")
                    lastError = error
                }
                group.leave()
            }
        }
        
        group.notify(queue: .main) {
            completion(lastError)
        }
    }
    
    // Helper function to generate a random US phone number
    private func generateRandomPhoneNumber() -> String {
        let areaCode = Int.random(in: 200...999)
        let prefix = Int.random(in: 200...999)
        let lineNumber = Int.random(in: 1000...9999)
        return "(\(areaCode)) \(prefix)-\(lineNumber)"
    }
    
    // MARK: - Image Storage
    
    // Note: In a real app, you would use Firebase Storage to save profile and property images
    // For now, we'll just record that they exist in the data
    
    // MARK: - Testing
    
    /// A method to test if Firebase is properly connected
    func testFirebaseConnection(completion: @escaping (Bool, Error?) -> Void) {
        let testData: [String: Any] = ["test": "data", "timestamp": FieldValue.serverTimestamp()]
        
        db.collection("test").document("connection").setData(testData) { error in
            if let error = error {
                completion(false, error)
            } else {
                completion(true, nil)
            }
        }
    }
} 