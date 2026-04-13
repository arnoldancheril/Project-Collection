//
//  PropertyPreferencesView.swift
//  RoommateSwipe
//

import SwiftUI

struct PropertyPreferencesView: View {
    @Binding var userData: UserRegistrationData
    @State private var newNeighborhood = ""
    @State private var selectedAmenities = Set<String>()
    
    let chicagoNeighborhoods = [
        "Wicker Park", "Logan Square", "Lincoln Park",
        "Lakeview", "River North", "West Loop",
        "Bucktown", "Old Town", "Gold Coast",
        "Ukrainian Village", "Pilsen", "Hyde Park"
    ]
    
    let amenities = [
        "In-unit Laundry", "Dishwasher", "Central AC",
        "Parking", "Gym", "Pool", "Elevator",
        "Doorman", "Roof Deck", "Storage",
        "Pet Friendly", "Furnished"
    ]
    
    var body: some View {
        VStack(spacing: 25) {
            Text("Property Preferences")
                .font(.title2)
                .fontWeight(.bold)
            
            ScrollView {
                VStack(alignment: .leading, spacing: 25) {
                    // Preferred Neighborhoods
                    VStack(alignment: .leading) {
                        Text("Preferred Neighborhoods")
                            .fontWeight(.medium)
                        
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack {
                                ForEach(chicagoNeighborhoods, id: \.self) { neighborhood in
                                    Toggle(neighborhood, isOn: Binding(
                                        get: { userData.preferredNeighborhoods.contains(neighborhood) },
                                        set: { isSelected in
                                            if isSelected {
                                                userData.preferredNeighborhoods.append(neighborhood)
                                            } else {
                                                userData.preferredNeighborhoods.removeAll { $0 == neighborhood }
                                            }
                                        }
                                    ))
                                    .toggleStyle(.button)
                                    .buttonStyle(.bordered)
                                }
                            }
                        }
                    }
                    
                    // Room Preference
                    VStack(alignment: .leading) {
                        Text("Room Preference")
                            .fontWeight(.medium)
                        
                        Picker("Room Type", selection: $userData.roomPreference) {
                            Text("Private Room").tag("Private")
                            Text("Shared Room").tag("Shared")
                            Text("Entire Unit").tag("Entire")
                        }
                        .pickerStyle(.segmented)
                    }
                    
                    // Desired Amenities
                    VStack(alignment: .leading) {
                        Text("Desired Amenities")
                            .fontWeight(.medium)
                        
                        LazyVGrid(columns: [
                            GridItem(.flexible()),
                            GridItem(.flexible())
                        ], spacing: 10) {
                            ForEach(amenities, id: \.self) { amenity in
                                Toggle(amenity, isOn: Binding(
                                    get: { userData.desiredAmenities.contains(amenity) },
                                    set: { isSelected in
                                        if isSelected {
                                            userData.desiredAmenities.append(amenity)
                                        } else {
                                            userData.desiredAmenities.removeAll { $0 == amenity }
                                        }
                                    }
                                ))
                                .toggleStyle(.button)
                                .buttonStyle(.bordered)
                            }
                        }
                    }
                    
                    // Additional Preferences
                    VStack(alignment: .leading) {
                        Text("Additional Preferences")
                            .fontWeight(.medium)
                        
                        TextEditor(text: .constant(""))
                            .frame(height: 100)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                            )
                    }
                }
                .padding()
            }
        }
    }
}

#Preview {
    PropertyPreferencesView(userData: .constant(UserRegistrationData()))
        .padding()
} 