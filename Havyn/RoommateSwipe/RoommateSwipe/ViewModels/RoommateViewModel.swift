//
//  RoommateViewModel.swift
//  RoommateSwipe
//
//  Created by AA on 2/27/25.
//

import SwiftUI

class RoommateViewModel: ObservableObject {
    // Example list of profiles for swiping
    @Published var profiles: [Profile] = [
        Profile(name: "Buzz",  age: 22, city: "Seattle", bio: "Surfer, thrift store diva.", imageName: "exampleProfile1"),
        Profile(name: "Shrek",  age: 24, city: "New York",      bio: "Looking for a long-term partner in crime.", imageName: "exampleProfile2"),
        Profile(name: "Scooby",   age: 25, city: "Chicago",       bio: "Tech enthusiast, coffee lover.", imageName: "exampleProfile3"),
        Profile(name: "Aladdin", age: 22, city: "San Francisco",       bio: "Dog person, weekend hiker.", imageName: "exampleProfile4")
    ]
    
    // Keep track of profiles you've liked
    @Published var likedProfiles: [Profile] = []
    
    // Keep track of mutual matches
    @Published var matchedProfiles: [Profile] = []
    
    // The current user’s info
    @Published var currentUser = UserProfile(
        name: "Your Name",
        age: 25,
        city: "Unknown",
        budget: "",
        interests: "",
        bio: "Tell us about yourself!"
    )
    
    // Called when the user swipes right on a profile
    func like(profile: Profile) {
        // Add to liked list
        likedProfiles.append(profile)
        
        // In a real app, you’d check if that profile also liked you.
        // If so, move to matchedProfiles. For demonstration:
        if Bool.random() {
            // Simulate that the other user liked you back
            matchedProfiles.append(profile)
        }
    }
    
    // Called when the user swipes left on a profile
    func dislike(profile: Profile) {
        // No action for now. Could store in a "passed" list if needed.
    }
    
    // Remove a profile from the main deck
    func removeProfile(profile: Profile) {
        profiles.removeAll { $0.id == profile.id }
    }
}
