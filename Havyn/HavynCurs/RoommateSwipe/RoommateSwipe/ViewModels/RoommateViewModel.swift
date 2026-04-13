//
//  RoommateViewModel.swift
//  RoommateSwipe
//

import SwiftUI
import CoreLocation
import Firebase
import FirebaseFirestore

class RoommateViewModel: ObservableObject {
    @Published var profiles: [Profile] = [
        Profile(
            name: "Elsa",
            age: 21,
            gender: "Female",
            city: "Chicago",
            bio: "Occasionally freezes the living room, loves to sing.",
            imageName: "exampleProfile5",
            propertyImageName: "exampleProperty5",
            hasRoom: true,
            needsRoom: false,
            moveInDate: Date().addingTimeInterval(30 * 24 * 60 * 60),
            preferredNeighborhoods: ["River North", "Gold Coast"],
            budgetRange: 800...1200,
            coordinate: CLLocationCoordinate2D(latitude: 41.8962, longitude: -87.6362), // River North
            numberOfRooms: 2,
            numberOfBathrooms: 1,
            amenities: "Private balcony, Modern decor, Fireplace",
            rent: "$1200 / month",
            address: "420 N State St, Chicago, IL 60654",
            cleanliness: 5,
            partying: 1,
            smoking: false,
            pets: false,
            petTypes: nil,
            wakeUpTime: "7:00 AM",
            sleepTime: "11:00 PM",
            habits: "Early riser, clean and organized, quiet after midnight.",
            lookingFor: "Someone who appreciates a well-maintained living space.",
            verificationStatus: true,
            isBlocked: false
        ),
        Profile(
            name: "Batman",
            age: 30,
            gender: "Male",
            city: "Chicago",
            bio: "Tech professional, enjoys city views and modern amenities.",
            imageName: "exampleProfile6",
            propertyImageName: "exampleProperty6",
            hasRoom: true,
            needsRoom: false,
            moveInDate: Date().addingTimeInterval(14 * 24 * 60 * 60),
            preferredNeighborhoods: ["Loop", "South Loop"],
            budgetRange: 4000...6000,
            coordinate: CLLocationCoordinate2D(latitude: 41.8786, longitude: -87.6251), // Loop
            numberOfRooms: 3,
            numberOfBathrooms: 2,
            amenities: "High-rise views, Gym, Doorman",
            rent: "$4500 / month",
            address: "300 E Randolph St, Chicago, IL 60601",
            cleanliness: 4,
            partying: 1,
            smoking: false,
            pets: false,
            petTypes: nil,
            wakeUpTime: "6:00 AM",
            sleepTime: "10:00 PM",
            habits: "Work from home, gym enthusiast, minimal cooking.",
            lookingFor: "Professional roommate with similar schedule.",
            verificationStatus: true,
            isBlocked: false
        ),
        Profile(
            name: "Homer",
            age: 39,
            gender: "Male",
            city: "Chicago",
            bio: "Loves donuts, craft beer enthusiast.",
            imageName: "exampleProfile7",
            propertyImageName: "exampleProperty7",
            hasRoom: true,
            needsRoom: false,
            moveInDate: Date().addingTimeInterval(7 * 24 * 60 * 60),
            preferredNeighborhoods: ["Wicker Park"],
            budgetRange: 800...1200,
            coordinate: CLLocationCoordinate2D(latitude: 41.9088, longitude: -87.6796), // Wicker Park
            numberOfRooms: 4,
            numberOfBathrooms: 2,
            amenities: "Rooftop deck, Garage parking, In-unit laundry",
            rent: "$1100 / month",
            address: "1550 N Milwaukee Ave, Chicago, IL 60622",
            cleanliness: 2,
            partying: 5,
            smoking: false,
            pets: true,
            petTypes: ["Dog"],
            wakeUpTime: "7:00 AM",
            sleepTime: "11:00 PM",
            habits: "Social, loves hosting game nights, casual living style.",
            lookingFor: "Someone who enjoys a laid-back atmosphere.",
            verificationStatus: true,
            isBlocked: false
        ),
        Profile(
            name: "Frodo",
            age: 28,
            gender: "Male",
            city: "Chicago",
            bio: "Artist and musician, looking for creative roommates.",
            imageName: "exampleProfile8",
            propertyImageName: "exampleProperty8",
            hasRoom: true,
            needsRoom: false,
            moveInDate: Date().addingTimeInterval(14 * 24 * 60 * 60),
            preferredNeighborhoods: ["Logan Square"],
            budgetRange: 600...800,
            coordinate: CLLocationCoordinate2D(latitude: 41.9231, longitude: -87.7093), // Logan Square
            numberOfRooms: 2,
            numberOfBathrooms: 1,
            amenities: "Music room, Art studio space, Garden",
            rent: "$750 / month",
            address: "2500 N Milwaukee Ave, Chicago, IL 60647",
            cleanliness: 3,
            partying: 3,
            smoking: false,
            pets: false,
            petTypes: nil,
            wakeUpTime: "9:00 AM",
            sleepTime: "1:00 AM",
            habits: "Night owl, creative projects, occasional band practice.",
            lookingFor: "Fellow artist or musician who appreciates creative energy.",
            verificationStatus: true,
            isBlocked: false
        ),
        Profile(
            name: "Buzz",
            age: 25,
            gender: "Female",
            city: "Chicago",
            bio: "Yoga instructor, plant enthusiast.",
            imageName: "exampleProfile1",
            propertyImageName: "exampleProperty1",
            hasRoom: true,
            needsRoom: false,
            moveInDate: Date().addingTimeInterval(30 * 24 * 60 * 60),
            preferredNeighborhoods: ["Lincoln Park"],
            budgetRange: 1000...1500,
            coordinate: CLLocationCoordinate2D(latitude: 41.9214, longitude: -87.6513), // Lincoln Park
            numberOfRooms: 2,
            numberOfBathrooms: 2,
            amenities: "Yoga space, Balcony garden, Natural light",
            rent: "$1300 / month",
            address: "2000 N Lincoln Park W, Chicago, IL 60614",
            cleanliness: 5,
            partying: 2,
            smoking: false,
            pets: true,
            petTypes: ["Cat"],
            wakeUpTime: "6:00 AM",
            sleepTime: "10:00 PM",
            habits: "Morning meditation, plant care, healthy cooking.",
            lookingFor: "Health-conscious roommate who enjoys a peaceful home.",
            verificationStatus: true,
            isBlocked: false
        ),
        Profile(
            name: "Shrek",
            age: 32,
            gender: "Male",
            city: "Chicago",
            bio: "Tech startup founder, coffee addict, fitness enthusiast.",
            imageName: "exampleProfile2",
            propertyImageName: "exampleProperty2",
            hasRoom: true,
            needsRoom: false,
            moveInDate: Date().addingTimeInterval(21 * 24 * 60 * 60),
            preferredNeighborhoods: ["West Loop", "Fulton Market"],
            budgetRange: 2000...3000,
            coordinate: CLLocationCoordinate2D(latitude: 41.8857, longitude: -87.6478), // West Loop
            numberOfRooms: 3,
            numberOfBathrooms: 2,
            amenities: "Home office, Smart home features, Fitness room",
            rent: "$2500 / month",
            address: "1000 W Randolph St, Chicago, IL 60607",
            cleanliness: 4,
            partying: 2,
            smoking: false,
            pets: false,
            petTypes: nil,
            wakeUpTime: "5:30 AM",
            sleepTime: "11:00 PM",
            habits: "Early morning workouts, works from home, loves cooking.",
            lookingFor: "Career-focused professional who values fitness and healthy living.",
            verificationStatus: true,
            isBlocked: false
        ),
        Profile(
            name: "Scooby",
            age: 27,
            gender: "Female",
            city: "Chicago",
            bio: "Freelance photographer, world traveler, foodie.",
            imageName: "exampleProfile3",
            propertyImageName: "exampleProperty3",
            hasRoom: true,
            needsRoom: false,
            moveInDate: Date().addingTimeInterval(45 * 24 * 60 * 60),
            preferredNeighborhoods: ["Bucktown", "Ukrainian Village"],
            budgetRange: 1200...1800,
            coordinate: CLLocationCoordinate2D(latitude: 41.9169, longitude: -87.6762), // Bucktown
            numberOfRooms: 2,
            numberOfBathrooms: 1,
            amenities: "Photography studio, Rooftop access, Vintage charm",
            rent: "$1600 / month",
            address: "1800 N Damen Ave, Chicago, IL 60647",
            cleanliness: 4,
            partying: 3,
            smoking: false,
            pets: true,
            petTypes: ["Cat"],
            wakeUpTime: "8:00 AM",
            sleepTime: "12:00 AM",
            habits: "Frequent traveler, home photography studio, loves hosting dinner parties.",
            lookingFor: "Creative individual who appreciates art and good food.",
            verificationStatus: true,
            isBlocked: false
        ),
        Profile(
            name: "Aladdin",
            age: 29,
            gender: "Male",
            city: "Chicago",
            bio: "Jazz musician, vinyl collector, culinary student.",
            imageName: "exampleProfile4",
            propertyImageName: "exampleProperty4",
            hasRoom: true,
            needsRoom: false,
            moveInDate: Date().addingTimeInterval(15 * 24 * 60 * 60),
            preferredNeighborhoods: ["Hyde Park", "Kenwood"],
            budgetRange: 900...1400,
            coordinate: CLLocationCoordinate2D(latitude: 41.7943, longitude: -87.5917), // Hyde Park
            numberOfRooms: 3,
            numberOfBathrooms: 2,
            amenities: "Music room, Chef's kitchen, Record collection space",
            rent: "$1200 / month",
            address: "5200 S Lake Shore Dr, Chicago, IL 60615",
            cleanliness: 3,
            partying: 2,
            smoking: false,
            pets: false,
            petTypes: nil,
            wakeUpTime: "9:00 AM",
            sleepTime: "1:00 AM",
            habits: "Late night practice sessions, cooking experiments, record collecting.",
            lookingFor: "Music lover who enjoys good food and late nights.",
            verificationStatus: true,
            isBlocked: false
        )
    ]

    // Current user's info
    @Published var currentUser = UserProfile(
        name: "Your Name",
        age: 25,
        city: "Unknown",
        budget: "",
        interests: "",
        bio: "Tell us about yourself!",
        email: "",
        phone: "",
        hasPlace: false,
        moveInDate: "",
        smoking: false,
        pets: false,
        wakeUpTime: "8:00 AM",
        sleepTime: "11:00 PM",
        cleanliness: 3,
        socialLevel: 3,
        imageName: "default_profile",
        profileImageName: nil,
        propertyImageName: nil
    )
    
    // A temporary user ID for testing - in a real app, this would come from authentication
    private let currentUserId = "testUser123"
    
    // Track loading and error states
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    // Lists of liked / matched profiles
    @Published var likedProfiles: [Profile] = []
    @Published var matchedProfiles: [Profile] = []
    
    // Filter properties
    @Published var filterCity: String = ""
    @Published var filterGender: String = "No Preference"
    @Published var filterBudget: ClosedRange<Double> = 0...10000
    @Published var filterMoveInDate: Date = Date().addingTimeInterval(90 * 24 * 60 * 60) // 90 days from now
    @Published var filterHasRoom: Bool = true
    @Published var filterNeedsRoom: Bool = false
    
    // Flag to track if profiles have been uploaded to Firebase
    @Published var sampleProfilesUploaded = false
    @Published var isUploadingSampleProfiles = false
    @Published var uploadError: String?
    
    // MARK: - Apartment Listing Data
    @Published var apartmentListings: [ApartmentListing] = []
    @Published var interestedUsers: [InterestedUser] = []
    
    // MARK: - Filter State
    @Published var filters = FilterOptions()
    
    // MARK: - Profile Management
    @Published var userProfile: Profile?
    @Published var editingProfile: Profile?
    
    // MARK: - Firestore Connection Test
    @Published var firebaseTestResult: Bool? = nil
    @Published var firebaseTestMessage: String = ""
    
    // MARK: - Firebase Service
    private let firebaseService = FirebaseService.shared
    
    init() {
        // Initialize the ViewModel
        print("RoommateViewModel initialized with \(profiles.count) sample profiles")
        
        // Try to load the user profile from Firestore when the app starts
        loadUserProfileFromFirestore()
    }
    
    // Upload sample profiles to Firebase
    func uploadSampleProfilesToFirebase() {
        guard !sampleProfilesUploaded && !isUploadingSampleProfiles else {
            // Don't upload if already uploaded or in progress
            return
        }
        
        isUploadingSampleProfiles = true
        
        FirebaseService.shared.uploadSampleProfiles(profiles: profiles) { [weak self] error in
            DispatchQueue.main.async {
                self?.isUploadingSampleProfiles = false
                
                if let error = error {
                    self?.uploadError = "Failed to upload profiles: \(error.localizedDescription)"
                    print("Error uploading sample profiles: \(error.localizedDescription)")
                } else {
                    self?.sampleProfilesUploaded = true
                    print("Successfully uploaded \(self?.profiles.count ?? 0) sample profiles to Firestore!")
                }
            }
        }
    }
    
    // Computed property for filtered profiles
    var filteredProfiles: [Profile] {
        profiles.filter { profile in
            // Filter by city if not empty
            if !filterCity.isEmpty && profile.city.lowercased() != filterCity.lowercased() {
                return false
            }
            
            // Filter by gender if specified
            if filterGender != "No Preference" && profile.gender != filterGender {
                return false
            }
            
            // Filter by budget
            if !profile.budgetRange.overlaps(filterBudget) {
                return false
            }
            
            // Filter by move-in date
            if profile.moveInDate > filterMoveInDate {
                return false
            }
            
            // Filter by room availability
            if filterHasRoom && !profile.hasRoom {
                return false
            }
            if filterNeedsRoom && !profile.needsRoom {
                return false
            }
            
            return true
        }
    }
    
    // MARK: - Firebase Integration
    
    /// Save the current user profile to Firestore
    func saveUserProfileToFirestore(completion: ((Error?) -> Void)? = nil) {
        isLoading = true
        errorMessage = nil
        
        FirebaseService.shared.saveUserProfile(currentUser, userId: currentUserId) { [weak self] error in
            DispatchQueue.main.async {
                self?.isLoading = false
                
                if let error = error {
                    self?.errorMessage = "Failed to save profile: \(error.localizedDescription)"
                    completion?(error)
                } else {
                    self?.errorMessage = nil
                    completion?(nil)
                }
            }
        }
    }
    
    /// Load the user profile from Firestore
    func loadUserProfileFromFirestore() {
        isLoading = true
        errorMessage = nil
        
        FirebaseService.shared.fetchUserProfile(userId: currentUserId) { [weak self] result in
            DispatchQueue.main.async {
                self?.isLoading = false
                
                switch result {
                case .success(let profile):
                    self?.currentUser = profile
                    self?.errorMessage = nil
                case .failure(let error):
                    // Only show error if it's not a "not found" error for new users
                    if (error as NSError).code != 404 {
                        self?.errorMessage = "Failed to load profile: \(error.localizedDescription)"
                    }
                }
            }
        }
    }
    
    /// Test the Firebase connection
    func testFirebaseConnection(completion: @escaping (Bool, String) -> Void) {
        let db = Firestore.firestore()
        let testDocument = db.collection("test").document("connection_test")
        
        let timestamp = Date().timeIntervalSince1970
        let testData: [String: Any] = ["timestamp": timestamp, "message": "Connection test"]
        
        testDocument.setData(testData) { error in
            if let error = error {
                self.firebaseTestResult = false
                self.firebaseTestMessage = "Failed to connect to Firebase: \(error.localizedDescription)"
                completion(false, self.firebaseTestMessage)
            } else {
                // Now try to read the data back
                testDocument.getDocument { (document, error) in
                    if let error = error {
                        self.firebaseTestResult = false
                        self.firebaseTestMessage = "Failed to read from Firebase: \(error.localizedDescription)"
                        completion(false, self.firebaseTestMessage)
                    } else if let document = document, document.exists {
                        self.firebaseTestResult = true
                        self.firebaseTestMessage = "Successfully connected to Firebase!"
                        completion(true, self.firebaseTestMessage)
                    } else {
                        self.firebaseTestResult = false
                        self.firebaseTestMessage = "Document doesn't exist after writing"
                        completion(false, self.firebaseTestMessage)
                    }
                }
            }
        }
    }
    
    // MARK: - Profile Actions
    
    // Basic actions
    func like(profile: Profile) {
        likedProfiles.append(profile)
        if Bool.random() {
            matchedProfiles.append(profile)
        }
    }
    
    func dislike(profile: Profile) { }
    
    func removeProfile(profile: Profile) {
        if let index = profiles.firstIndex(where: { $0.id == profile.id }) {
            profiles.remove(at: index)
        }
    }
    
    func blockProfile(profile: Profile) {
        if let index = profiles.firstIndex(where: { $0.id == profile.id }) {
            profiles.remove(at: index)
        }
    }
    
    func reportProfile(profile: Profile, reason: String) {
        // In a real app, you would send this to your backend
        print("Reported profile: \(profile.name) for reason: \(reason)")
    }
    
    // MARK: - Apartment Listing Methods
    
    func addApartmentListing(_ listing: ApartmentListing) {
        apartmentListings.append(listing)
        // TODO: Save to Firebase
    }
    
    func updateApartmentListing(_ listing: ApartmentListing) {
        if let index = apartmentListings.firstIndex(where: { $0.id == listing.id }) {
            apartmentListings[index] = listing
            // TODO: Update in Firebase
        }
    }
    
    func updateListing(_ listing: ApartmentListing) {
        // This is a convenience method that calls updateApartmentListing
        updateApartmentListing(listing)
    }
    
    func deleteApartmentListing(id: String) {
        apartmentListings.removeAll { $0.id == id }
        // TODO: Delete from Firebase
    }
    
    func toggleListingActive(id: String) {
        if let index = apartmentListings.firstIndex(where: { $0.id == id }) {
            apartmentListings[index].isActive.toggle()
            // TODO: Update in Firebase
        }
    }
    
    // MARK: - Interested User Methods
    
    func addInterestedUser(_ user: InterestedUser) {
        interestedUsers.append(user)
        
        // Update the listing to include this user's ID
        if let index = apartmentListings.firstIndex(where: { $0.id == user.listingId }) {
            if !apartmentListings[index].interestedUsers.contains(user.id) {
                apartmentListings[index].interestedUsers.append(user.id)
                // TODO: Update in Firebase
            }
        }
    }
    
    func updateInterestedUserStatus(userId: String, listingId: String, status: InterestStatus) {
        if let index = interestedUsers.firstIndex(where: { $0.id == userId && $0.listingId == listingId }) {
            interestedUsers[index].status = status
            // TODO: Update in Firebase
        }
    }
    
    func addMessageToConversation(userId: String, listingId: String, text: String, isFromLister: Bool) {
        if let index = interestedUsers.firstIndex(where: { $0.id == userId && $0.listingId == listingId }) {
            let newMessage = Message(
                id: UUID().uuidString,
                text: text,
                senderId: isFromLister ? "owner" : userId,
                timestamp: Date(),
                isFromLister: isFromLister
            )
            
            interestedUsers[index].messages.append(newMessage)
            
            // Update status to contacted if it was new
            if interestedUsers[index].status == .new {
                interestedUsers[index].status = .contacted
            }
            
            // TODO: Update in Firebase
        }
    }
}
