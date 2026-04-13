//
//  DetailedPropertyView.swift
//  RoommateSwipe
//

import SwiftUI
import MapKit

struct DetailedPropertyView: View {
    let profile: Profile
    
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Large property image
                    if let propertyImageName = profile.propertyImageName {
                        Image(propertyImageName)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: .infinity)
                            .clipped()
                    } else {
                        Rectangle()
                            .fill(Color.gray.opacity(0.4))
                            .frame(height: 300)
                            .overlay(Text("No property image").font(.headline).foregroundColor(.white))
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Property Info")
                            .font(.title)
                            .fontWeight(.bold)
                            .padding(.horizontal)
                        
                        // Room Information
                        HStack {
                            VStack(alignment: .leading) {
                                Text("Rooms")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                Text("\(profile.numberOfRooms)")
                                    .font(.headline)
                            }
                            .frame(maxWidth: .infinity)
                            
                            VStack(alignment: .leading) {
                                Text("Bathrooms")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                Text("\(profile.numberOfBathrooms)")
                                    .font(.headline)
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(10)
                        .padding(.horizontal)

                        // Amenities
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Amenities")
                                .font(.headline)
                            Text(profile.amenities)
                                .font(.body)
                        }
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(10)
                        .padding(.horizontal)

                        // Rent
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Monthly Rent")
                                .font(.headline)
                            Text(profile.rent)
                                .font(.title2)
                                .foregroundColor(.blue)
                        }
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(10)
                        .padding(.horizontal)
                    }
                }
            }
            .navigationTitle("Property Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") {
                        dismiss()
                    }
                }
            }
        }
    }
}

#Preview {
    DetailedPropertyView(
        profile: Profile(
            name: "Aladdin",
            age: 22,
            gender: "Male",
            city: "Chicago",
            bio: "Dog person, weekend hiker.",
            imageName: "exampleProfile4",
            propertyImageName: "exampleProperty4",
            hasRoom: true,
            needsRoom: false,
            moveInDate: Date().addingTimeInterval(30 * 24 * 60 * 60), // 30 days from now
            preferredNeighborhoods: ["Wicker Park", "Logan Square"],
            budgetRange: 1000...1500,
            coordinate: CLLocationCoordinate2D(latitude: 41.9088, longitude: -87.6796), // Wicker Park
            numberOfRooms: 2,
            numberOfBathrooms: 2,
            amenities: "Covered parking, Storage",
            rent: "$1100 / month",
            address: "1550 N Milwaukee Ave, Chicago, IL 60622",
            cleanliness: 4,
            partying: 2,
            smoking: false,
            pets: true,
            petTypes: ["Abu"],
            wakeUpTime: "6:00 AM",
            sleepTime: "10:00 PM",
            habits: "Clean, early riser, occasionally blasts Disney tunes.",
            lookingFor: "Friendly, active roommate who loves the outdoors.",
            verificationStatus: true,
            isBlocked: false
        )
    )
}
