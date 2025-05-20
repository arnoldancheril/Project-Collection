//
//  Profile.swift
//  RoommateSwipe
//
//  Created by Arnold Ancheril on 2/27/25.
//

import SwiftUI

struct Profile: Identifiable {
    let id = UUID()
    let name: String
    let age: Int
    let city: String
    let bio: String
    let imageName: String // The name of the image asset or system name

    // Example additional fields:
    // let budget: String
    // let hobbies: [String]
    // etc.
}
