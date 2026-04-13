//
//  Profile.swift
//  RoommateSwipe
//

import SwiftUI
import MapKit

struct Profile: Identifiable {
    let id = UUID()
    
    // Basic Info
    let name: String
    let age: Int
    let gender: String
    let city: String
    let bio: String
    
    // Profile Images
    let imageName: String
    let propertyImageName: String?
    
    // Living Situation
    let hasRoom: Bool
    let needsRoom: Bool
    let moveInDate: Date
    
    // Location & Budget
    let preferredNeighborhoods: [String]
    let budgetRange: ClosedRange<Double>
    let coordinate: CLLocationCoordinate2D
    
    // Property Details (if hasRoom is true)
    let numberOfRooms: Int
    let numberOfBathrooms: Int
    let amenities: String
    let rent: String
    let address: String
    
    // Lifestyle Preferences
    let cleanliness: Int // 1-5 scale
    let partying: Int // 1-5 scale
    let smoking: Bool
    let pets: Bool
    let petTypes: [String]?
    let wakeUpTime: String
    let sleepTime: String
    
    // Additional Info
    let habits: String
    let lookingFor: String
    let verificationStatus: Bool
    let isBlocked: Bool
    
    // Computed property for profile type
    var profileType: ProfileType {
        if hasRoom && needsRoom {
            return .both
        } else if hasRoom {
            return .offering
        } else if needsRoom {
            return .seeking
        } else {
            return .apartmentCompany
        }
    }
}

enum ProfileType: String {
    case offering = "offering"
    case seeking = "seeking"
    case both = "both"
    case apartmentCompany = "apartmentCompany"
}
