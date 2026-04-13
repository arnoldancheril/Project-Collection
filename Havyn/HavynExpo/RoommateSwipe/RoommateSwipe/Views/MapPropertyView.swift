//
//  MapPropertyView.swift
//  RoommateSwipe
//

import SwiftUI
import MapKit

struct MapPropertyView: View {
    let profile: Profile
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Property Image
                if let propertyImageName = profile.propertyImageName {
                    Image(propertyImageName)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity)
                        .clipped()
                }
                
                VStack(alignment: .leading, spacing: 16) {
                    // Property Info Section
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Property Details")
                            .font(.title)
                            .fontWeight(.bold)
                        
                        Text(profile.address)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        Text(profile.rent)
                            .font(.title2)
                            .fontWeight(.semibold)
                            .foregroundColor(.blue)
                    }
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
                    
                    Divider()
                        .padding(.vertical)
                    
                    // Profile Info Section
                    VStack(alignment: .leading, spacing: 16) {
                        HStack(spacing: 16) {
                            Image(profile.imageName)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 80, height: 80)
                                .clipShape(Circle())
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text("\(profile.name), \(profile.age)")
                                    .font(.title2)
                                    .fontWeight(.bold)
                                Text(profile.city)
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(.horizontal)
                        
                        Text(profile.bio)
                            .font(.body)
                            .padding(.horizontal)
                        
                        // Lifestyle Section
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Lifestyle")
                                .font(.headline)
                                .padding(.bottom, 4)
                            
                            HStack {
                                LifestyleTag(title: "Cleanliness", value: profile.cleanliness)
                                LifestyleTag(title: "Partying", value: profile.partying)
                            }
                            
                            HStack {
                                if profile.smoking {
                                    LifestyleTag(title: "Smoker", isToggle: true)
                                }
                                if profile.pets {
                                    LifestyleTag(title: "Has Pets", isToggle: true)
                                }
                            }
                            
                            Text("Schedule: \(profile.wakeUpTime) - \(profile.sleepTime)")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(10)
                        .padding(.horizontal)
                    }
                }
            }
            .padding(.bottom, 20)
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button("Back") {
                    dismiss()
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Contact") {
                    // Implement contact functionality
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }
}

struct LifestyleTag: View {
    let title: String
    var value: Int? = nil
    var isToggle: Bool = false
    
    var body: some View {
        HStack {
            Text(title)
            if let value = value {
                Text(String(repeating: "•", count: value))
                    .foregroundColor(.blue)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.blue.opacity(0.1))
        .cornerRadius(15)
    }
}

#Preview {
    NavigationView {
        MapPropertyView(
            profile: Profile(
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
                coordinate: CLLocationCoordinate2D(latitude: 41.8962, longitude: -87.6362),
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
            )
        )
    }
} 