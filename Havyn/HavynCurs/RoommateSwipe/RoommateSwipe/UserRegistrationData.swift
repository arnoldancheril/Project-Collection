//
//  UserRegistrationData.swift
//  RoommateSwipe
//

import SwiftUI
import UIKit

struct UserRegistrationData {
    var firstName = ""
    var lastName = ""
    var email = ""
    var password = ""
    var dateOfBirth = Date()
    var gender = ""
    var occupation = ""
    var bio = ""
    
    // Lifestyle
    var cleanliness = 3
    var noise = 3
    var socialLevel = 3
    var sleepSchedule = "Night Owl"
    var drinking = false
    var smoking = false
    var pets = false
    
    // Preferences
    var preferredGender = "Any"
    var ageRange = 18...35
    var rentRange = 500...3000
    var moveInDate = Date()
    
    // Schedule
    var wakeUpTime = Date()
    var bedTime = Date()
    var workSchedule = "9-5"
    
    // Property
    var preferredNeighborhoods: [String] = []
    var desiredAmenities: [String] = []
    var roomPreference = "Private"
    
    // Photos
    var profilePhoto: UIImage?
    var propertyPhotos: [UIImage] = []
} 