//
//  ProfileView.swift
//  RoommateSwipe
//

import SwiftUI

struct ProfileView: View {
    @EnvironmentObject var viewModel: RoommateViewModel
    
    // Local state to reflect the currentUser's fields
    @State private var hasPlace: Bool = false
    @State private var genderPreference: String = "No Preference" // "Male", "Female", or "No Preference"
    @State private var cityPreference: String = ""
    @State private var desiredAmenities: String = ""

    var body: some View {
        NavigationView {
            Form {
                // SECTION 1: Profile Images
                Section(header: Text("Profile Images")) {
                    // Show current user profile image if available
                    if let imageName = viewModel.currentUser.profileImageName,
                       let userImage = UIImage(named: imageName) {
                        Image(uiImage: userImage)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 200)
                            .cornerRadius(12)
                    } else {
                        Text("No profile image yet")
                            .foregroundColor(.secondary)
                    }
                    
                    Button("Add/Update Profile Image") {
                        // In a real app, open a UIImagePicker or PhotosPicker
                        print("User taps to pick or update profile image.")
                    }
                }
                
                // SECTION 2: Property Images (if user has a place)
                Section(header: Text("Property Images")) {
                    Toggle("I already have a place", isOn: $hasPlace)
                    
                    if hasPlace {
                        if let propImageName = viewModel.currentUser.propertyImageName,
                           let propertyImage = UIImage(named: propImageName) {
                            Image(uiImage: propertyImage)
                                .resizable()
                                .scaledToFit()
                                .frame(height: 200)
                                .cornerRadius(12)
                        } else {
                            Text("No property image yet")
                                .foregroundColor(.secondary)
                        }
                        
                        Button("Add/Update Property Image") {
                            print("Open image picker for property photo.")
                        }
                    } else {
                        Text("You currently don't have a place.")
                            .foregroundColor(.secondary)
                    }
                }
                
                // SECTION 3: House/Room Preferences
                Section(header: Text("Looking For")) {
                    if hasPlace {
                        TextField("Desired Amenities", text: $desiredAmenities)
                            .textFieldStyle(.roundedBorder)
                            .placeholder(when: desiredAmenities.isEmpty) {
                                Text("e.g. Washer/Dryer, Gym, etc.")
                                    .foregroundColor(.gray)
                            }
                    } else {
                        Text("Since you don't have a place, you can specify what type of place you're looking for below.")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                }
                
                // SECTION 4: Search / Filter Preferences
                Section(header: Text("Filter Preferences")) {
                    TextField("City Preference", text: $cityPreference)
                        .textFieldStyle(.roundedBorder)
                        .placeholder(when: cityPreference.isEmpty) {
                            Text("e.g. New York, Chicago")
                                .foregroundColor(.gray)
                        }
                    
                    Picker("Gender Preference", selection: $genderPreference) {
                        Text("No Preference").tag("No Preference")
                        Text("Male").tag("Male")
                        Text("Female").tag("Female")
                    }
                    .pickerStyle(.segmented)
                }
                
                // SECTION 5: Save Button
                Section {
                    Button("Save") {
                        // Example: update your model
                        // viewModel.currentUser.profileImageName = ...
                        // viewModel.currentUser.propertyImageName = ...
                        // Save filter preferences, etc.
                        print("Profile saved! (Placeholder)")
                    }
                }
            }
            .navigationTitle("My Profile")
        }
        .onAppear {
            // Initialize local state from the environment object's currentUser
            // If you want them to persist across sessions, store them in currentUser or in a backend
            // For example:
            // hasPlace = (viewModel.currentUser.propertyImageName != nil)
            // cityPreference = viewModel.currentUser.city
        }
    }
}

extension View {
    // Simple placeholder extension
    func placeholder<Content: View>(
        when shouldShow: Bool,
        alignment: Alignment = .leading,
        @ViewBuilder placeholder: () -> Content
    ) -> some View {
        ZStack(alignment: alignment) {
            if shouldShow {
                placeholder()
            }
            self
        }
    }
}

#Preview {
    ProfileView()
        .environmentObject(RoommateViewModel())
}
