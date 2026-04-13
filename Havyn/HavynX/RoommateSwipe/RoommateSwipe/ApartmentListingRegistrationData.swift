import SwiftUI
import UIKit

struct ApartmentListingRegistrationData {
    // Company Information
    var companyName = ""
    var contactPersonName = ""
    var email = ""
    var password = ""
    var contactPhoneNumber = ""
    
    // Property Details
    var city = ""
    var neighborhood = ""
    var apartmentName = ""
    var address = ""
    var numberOfAvailableRooms = 1
    var availabilityDate = Date()
    var minPrice = 1000
    var maxPrice = 2000
    
    // Property Features
    var housingType = "Apartment" // Apartment, Condo, House
    var amenities: [String] = []
    var isPetFriendly = false
    var isSmokingAllowed = false
    
    // Visuals
    var propertyDescription = ""
    var propertyPhotos: [UIImage] = []
} 