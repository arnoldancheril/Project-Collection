import SwiftUI
import UIKit

struct HaveRoomRegistrationData {
    // Basic Information
    var fullName = ""
    var email = ""
    var password = ""
    var phoneNumber = ""
    var dateOfBirth = Date()
    
    // About Your Space
    var city = ""
    var neighborhood = ""
    var moveInDate = Date()
    var rentPrice = 1000
    var housingType = "Apartment" // Apartment, House, Shared Space
    var isFurnished = false
    
    // Roommate Preferences
    var genderPreference = "No Preference" // Male, Female, Non-binary, No Preference
    var smokingPreference = false // Allow smoking
    var petPreference = false // Allow pets
    var lifestylePreference = "Flexible" // Morning person, Night owl, Flexible
    var cleanlinessLevel = 3 // 1-5 scale
    
    // Personalization and Interests
    var bio = ""
    var interests: [String] = []
    var profilePhoto: UIImage?
    var roomPhotos: [UIImage] = []
} 