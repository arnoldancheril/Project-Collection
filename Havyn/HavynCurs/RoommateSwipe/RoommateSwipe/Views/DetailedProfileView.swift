//
//  DetailedProfileView.swift
//  RoommateSwipe
//

import SwiftUI
import MapKit

struct DetailedProfileView: View {
    let profile: Profile
    
    // Callbacks that let us "like" or "dislike" this profile
    let onLike: () -> Void
    let onDislike: () -> Void

    // Used to dismiss the sheet
    @Environment(\.dismiss) private var dismiss

    // Show a second sheet with more property details
    @State private var showPropertySheet = false

    // State for animations
    @State private var showingLikeFeedback = false
    @State private var showingPassFeedback = false

    // Function to show feedback animation based on action
    private func showFeedbackAnimation(isLike: Bool) {
        if isLike {
            showingLikeFeedback = true
            // Reset after animation completes
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                showingLikeFeedback = false
            }
        } else {
            showingPassFeedback = true
            // Reset after animation completes
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                showingPassFeedback = false
            }
        }
    }

    var body: some View {
        NavigationView {
            ZStack(alignment: .center) {
                ScrollView(.vertical, showsIndicators: true) {
                    VStack(alignment: .leading, spacing: 20) {
                        // Profile Image
                        Image(profile.imageName)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: .infinity)
                            .clipped()

                        VStack(alignment: .leading, spacing: 16) {
                            // Basic Info
                            VStack(alignment: .leading, spacing: 8) {
                                Text("\(profile.name), \(profile.age)")
                                    .font(.title)
                                    .fontWeight(.bold)

                                Text(profile.city)
                                    .font(.title3)
                                    .foregroundColor(.secondary)

                                Text(profile.bio)
                                    .font(.body)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .padding(.top, 4)
                            }
                            .padding(.horizontal)

                            Divider()
                                .padding(.horizontal)

                            // Habits Section
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Habits")
                                    .font(.title2)
                                    .fontWeight(.semibold)
                                Text(profile.habits)
                                    .font(.body)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .padding(.horizontal)

                            Divider()
                                .padding(.horizontal)

                            // Looking For Section
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Looking For")
                                    .font(.title2)
                                    .fontWeight(.semibold)
                                Text(profile.lookingFor)
                                    .font(.body)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .padding(.horizontal)

                            Divider()
                                .padding(.horizontal)

                            // Home Details Section
                            VStack(alignment: .leading, spacing: 12) {
                                Text("Home Details")
                                    .font(.title2)
                                    .fontWeight(.semibold)

                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Rooms: \(profile.numberOfRooms)")
                                    Text("Bathrooms: \(profile.numberOfBathrooms)")
                                    Text("Rent: \(profile.rent)")
                                        .foregroundColor(.blue)
                                    Text("Amenities: \(profile.amenities)")
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                                .font(.body)
                            }
                            .padding(.horizontal)

                            // Property Info Button
                            Button {
                                showPropertySheet = true
                            } label: {
                                Text("View Property Details")
                                    .fontWeight(.semibold)
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(Color.blue.opacity(0.2))
                                    .cornerRadius(10)
                            }
                            .padding(.horizontal)
                            .padding(.top, 8)
                        }
                    }
                }
                .navigationTitle("Profile Details")
                .navigationBarTitleDisplayMode(.inline)
                .navigationBarBackButtonHidden(true)
                .navigationBarItems(leading: Button("Back") {
                    dismiss()
                })
                .safeAreaInset(edge: .bottom) {
                    HStack(spacing: 12) {
                        Button {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                // Add animation showing feedback
                                showFeedbackAnimation(isLike: false)
                            }
                            
                            // Delay closing the sheet slightly to show animation
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                onDislike()  // Mark as Not Interested
                                dismiss()    // Close the sheet
                            }
                        } label: {
                            HStack {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.system(size: 20))
                                Text("Not Interested")
                                    .fontWeight(.semibold)
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .foregroundColor(.white)
                            .background(
                                LinearGradient(
                                    gradient: Gradient(colors: [Color.red.opacity(0.7), Color.red]),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .cornerRadius(10)
                            .shadow(color: Color.red.opacity(0.3), radius: 3, x: 0, y: 2)
                        }
                        .buttonStyle(ScaleButtonStyle())

                        Button {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                // Add animation showing feedback
                                showFeedbackAnimation(isLike: true)
                            }
                            
                            // Delay closing the sheet slightly to show animation
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                onLike()     // Mark as Like
                                dismiss()    // Close the sheet
                            }
                        } label: {
                            HStack {
                                Image(systemName: "heart.fill")
                                    .font(.system(size: 20))
                                Text("Like")
                                    .fontWeight(.semibold)
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .foregroundColor(.white)
                            .background(
                                LinearGradient(
                                    gradient: Gradient(colors: [Color.green.opacity(0.7), Color.green]),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .cornerRadius(10)
                            .shadow(color: Color.green.opacity(0.3), radius: 3, x: 0, y: 2)
                        }
                        .buttonStyle(ScaleButtonStyle())
                    }
                    .padding()
                    .background(.thinMaterial)
                }
                
                // Feedback indicators
                if showingLikeFeedback {
                    VStack {
                        Text("LIKE")
                            .font(.system(size: 48, weight: .heavy))
                            .foregroundColor(.green)
                            .padding(20)
                            .background(Color.white.opacity(0.7))
                            .cornerRadius(15)
                            .overlay(
                                RoundedRectangle(cornerRadius: 15)
                                    .stroke(Color.green, lineWidth: 5)
                            )
                            .rotationEffect(.degrees(-15))
                            .shadow(color: Color.green.opacity(0.5), radius: 10, x: 0, y: 5)
                    }
                    .transition(.scale(scale: 0.8).combined(with: .opacity))
                    .animation(.spring(response: 0.3, dampingFraction: 0.6), value: showingLikeFeedback)
                }
                
                if showingPassFeedback {
                    VStack {
                        Text("PASS")
                            .font(.system(size: 48, weight: .heavy))
                            .foregroundColor(.red)
                            .padding(20)
                            .background(Color.white.opacity(0.7))
                            .cornerRadius(15)
                            .overlay(
                                RoundedRectangle(cornerRadius: 15)
                                    .stroke(Color.red, lineWidth: 5)
                            )
                            .rotationEffect(.degrees(15))
                            .shadow(color: Color.red.opacity(0.5), radius: 10, x: 0, y: 5)
                    }
                    .transition(.scale(scale: 0.8).combined(with: .opacity))
                    .animation(.spring(response: 0.3, dampingFraction: 0.6), value: showingPassFeedback)
                }
            }
            .sheet(isPresented: $showPropertySheet) {
                // REUSE the DetailedPropertyView
                DetailedPropertyView(profile: profile)
            }
        }
    }
}

#Preview {
    DetailedProfileView(
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
            habits: "I wake up early, keep common areas clean, no smoking.",
            lookingFor: "A tidy roommate who respects quiet hours.",
            verificationStatus: true,
            isBlocked: false
        ),
        onLike: { },
        onDislike: { }
    )
}
