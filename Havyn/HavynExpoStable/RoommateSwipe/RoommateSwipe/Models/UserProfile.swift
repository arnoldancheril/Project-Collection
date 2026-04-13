//
//  UserProfile.swift
//  RoommateSwipe
//

import Foundation

struct UserProfile {
    var name: String
    var age: Int
    var city: String
    var budget: String
    var interests: String
    var bio: String
    var email: String
    var phone: String
    var hasPlace: Bool
    var moveInDate: String
    var smoking: Bool
    var pets: Bool
    var wakeUpTime: String
    var sleepTime: String
    var cleanliness: Int // 1-5 scale
    var socialLevel: Int // 1-5 scale
    var imageName: String
    
    // Optional images for user + property
    var profileImageName: String?
    var propertyImageName: String?
    
    // Initialize with default values
    init(name: String = "",
         age: Int = 0,
         city: String = "",
         budget: String = "",
         interests: String = "",
         bio: String = "",
         email: String = "",
         phone: String = "",
         hasPlace: Bool = false,
         moveInDate: String = "",
         smoking: Bool = false,
         pets: Bool = false,
         wakeUpTime: String = "8:00 AM",
         sleepTime: String = "11:00 PM",
         cleanliness: Int = 3,
         socialLevel: Int = 3,
         imageName: String = "default_profile",
         profileImageName: String? = nil,
         propertyImageName: String? = nil) {
        self.name = name
        self.age = age
        self.city = city
        self.budget = budget
        self.interests = interests
        self.bio = bio
        self.email = email
        self.phone = phone
        self.hasPlace = hasPlace
        self.moveInDate = moveInDate
        self.smoking = smoking
        self.pets = pets
        self.wakeUpTime = wakeUpTime
        self.sleepTime = sleepTime
        self.cleanliness = cleanliness
        self.socialLevel = socialLevel
        self.imageName = imageName
        self.profileImageName = profileImageName
        self.propertyImageName = propertyImageName
    }
}
