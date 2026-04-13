//
//  ProfileEditView.swift
//  RoommateSwipe
//

import SwiftUI
import PhotosUI

struct ProfileEditView: View {
    @EnvironmentObject var viewModel: RoommateViewModel
    @Environment(\.dismiss) private var dismiss
    
    // Local state to reflect user input
    @State private var name: String = ""
    @State private var email: String = ""
    @State private var phone: String = ""
    @State private var age: Int = 0
    @State private var city: String = ""
    @State private var hasPlace: Bool = false
    @State private var moveInDate: String = ""
    @State private var budget: String = ""
    @State private var smoking: Bool = false
    @State private var pets: Bool = false
    @State private var wakeUpTime: String = ""
    @State private var sleepTime: String = ""
    @State private var cleanliness: Int = 3
    @State private var socialLevel: Int = 3
    @State private var bio: String = ""
    @State private var interests: String = ""
    
    // For image selection
    @State private var selectedProfileItem: PhotosPickerItem?
    @State private var selectedPropertyItem: PhotosPickerItem?
    @State private var profileImage: UIImage?
    @State private var propertyImage: UIImage?
    
    // For showing alerts and loading indicators
    @State private var showingSaveSuccessAlert = false
    @State private var showingSaveErrorAlert = false
    @State private var isSaving = false
    
    var body: some View {
        NavigationView {
            ZStack {
                Form {
                    // Profile Image Section
                    Section(header: Text("Profile Photo")) {
                        HStack {
                            Spacer()
                            if let profileImage = profileImage {
                                Image(uiImage: profileImage)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 120, height: 120)
                                    .clipShape(Circle())
                                    .overlay(Circle().stroke(Color.white, lineWidth: 4))
                                    .shadow(radius: 5)
                            } else if let imageName = viewModel.currentUser.profileImageName,
                                    let image = UIImage(named: imageName) {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 120, height: 120)
                                    .clipShape(Circle())
                                    .overlay(Circle().stroke(Color.white, lineWidth: 4))
                                    .shadow(radius: 5)
                            } else {
                                Image(systemName: "person.circle.fill")
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: 120, height: 120)
                                    .foregroundColor(.gray)
                                    .overlay(Circle().stroke(Color.white, lineWidth: 4))
                                    .shadow(radius: 5)
                            }
                            Spacer()
                        }
                        .padding(.vertical, 8)
                        
                        PhotosPicker("Select Profile Photo", selection: $selectedProfileItem, matching: .images)
                            .onChange(of: selectedProfileItem) { newItem in
                                Task {
                                    if let data = try? await newItem?.loadTransferable(type: Data.self),
                                       let uiImage = UIImage(data: data) {
                                        profileImage = uiImage
                                    }
                                }
                            }
                    }
                    
                    // Basic Information Section
                    Section(header: Text("Basic Information")) {
                        TextField("Name", text: $name)
                        TextField("Email", text: $email)
                            .keyboardType(.emailAddress)
                            .autocapitalization(.none)
                        TextField("Phone", text: $phone)
                            .keyboardType(.phonePad)
                        
                        Stepper(value: $age, in: 18...100) {
                            HStack {
                                Text("Age")
                                Spacer()
                                Text("\(age)")
                            }
                        }
                    }
                    
                    // Housing Preferences Section
                    Section(header: Text("Housing Preferences")) {
                        Toggle("I have a place", isOn: $hasPlace)
                        
                        TextField("City", text: $city)
                        TextField("Move-in Date", text: $moveInDate)
                        TextField("Budget", text: $budget)
                            .keyboardType(.decimalPad)
                        
                        if hasPlace {
                            HStack {
                                Spacer()
                                if let propertyImage = propertyImage {
                                    Image(uiImage: propertyImage)
                                        .resizable()
                                        .scaledToFill()
                                        .frame(height: 150)
                                        .clipShape(RoundedRectangle(cornerRadius: 12))
                                        .shadow(radius: 5)
                                } else if let propertyImageName = viewModel.currentUser.propertyImageName,
                                        let image = UIImage(named: propertyImageName) {
                                    Image(uiImage: image)
                                        .resizable()
                                        .scaledToFill()
                                        .frame(height: 150)
                                        .clipShape(RoundedRectangle(cornerRadius: 12))
                                        .shadow(radius: 5)
                                } else {
                                    Text("No property image")
                                        .frame(height: 150)
                                        .frame(maxWidth: .infinity)
                                        .background(Color.gray.opacity(0.2))
                                        .clipShape(RoundedRectangle(cornerRadius: 12))
                                }
                                Spacer()
                            }
                            .padding(.vertical, 8)
                            
                            PhotosPicker("Select Property Photo", selection: $selectedPropertyItem, matching: .images)
                                .onChange(of: selectedPropertyItem) { newItem in
                                    Task {
                                        if let data = try? await newItem?.loadTransferable(type: Data.self),
                                           let uiImage = UIImage(data: data) {
                                            propertyImage = uiImage
                                        }
                                    }
                                }
                        }
                    }
                    
                    // Lifestyle Section
                    Section(header: Text("Lifestyle & Compatibility")) {
                        Toggle("Smoking", isOn: $smoking)
                        Toggle("Pets", isOn: $pets)
                        
                        TextField("Wake-up Time (e.g., 7:00 AM)", text: $wakeUpTime)
                        TextField("Sleep Time (e.g., 11:00 PM)", text: $sleepTime)
                        
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Cleanliness Level")
                            HStack {
                                Text("Relaxed")
                                    .font(.caption)
                                Slider(value: .init(get: { Double(cleanliness) }, set: { cleanliness = Int($0) }), 
                                       in: 1...5, step: 1)
                                Text("Spotless")
                                    .font(.caption)
                            }
                            Text("Level: \(cleanlinessLevelText)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Social Level")
                            HStack {
                                Text("Private")
                                    .font(.caption)
                                Slider(value: .init(get: { Double(socialLevel) }, set: { socialLevel = Int($0) }), 
                                       in: 1...5, step: 1)
                                Text("Very Social")
                                    .font(.caption)
                            }
                            Text("Level: \(socialLevelText)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    // Bio & Interests Section
                    Section(header: Text("About Me")) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Bio")
                            TextEditor(text: $bio)
                                .frame(height: 100)
                                .padding(4)
                                .background(Color(.systemGray6))
                                .cornerRadius(8)
                        }
                        
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Interests (comma separated)")
                            TextEditor(text: $interests)
                                .frame(height: 80)
                                .padding(4)
                                .background(Color(.systemGray6))
                                .cornerRadius(8)
                            Text("Example: hiking, cooking, movies, reading")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    // Save Button Section
                    Section {
                        Button("Save Changes") {
                            saveChanges()
                        }
                        .frame(maxWidth: .infinity, alignment: .center)
                        .foregroundColor(.white)
                        .padding()
                        .background(isSaving ? Color.gray : Color.blue)
                        .cornerRadius(12)
                        .disabled(isSaving)
                    }
                }
                
                // Loading overlay
                if isSaving {
                    Color.black.opacity(0.4)
                        .ignoresSafeArea()
                        .overlay(
                            VStack {
                                ProgressView()
                                    .scaleEffect(1.5)
                                    .padding()
                                
                                Text("Saving profile...")
                                    .foregroundColor(.white)
                                    .fontWeight(.medium)
                                    .padding()
                            }
                            .padding(30)
                            .background(Color(.systemBackground).opacity(0.8))
                            .cornerRadius(12)
                            .shadow(radius: 10)
                        )
                }
            }
            .navigationTitle("Edit Profile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .disabled(isSaving)
                }
            }
            .alert("Profile Saved", isPresented: $showingSaveSuccessAlert) {
                Button("OK") {
                    dismiss()
                }
            } message: {
                Text("Your profile has been successfully updated.")
            }
            .alert("Error Saving Profile", isPresented: $showingSaveErrorAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                if let errorMessage = viewModel.errorMessage {
                    Text(errorMessage)
                } else {
                    Text("An unknown error occurred while saving your profile.")
                }
            }
        }
        .onAppear {
            loadUserData()
        }
    }
    
    private var cleanlinessLevelText: String {
        switch cleanliness {
        case 1: return "Relaxed"
        case 2: return "Casual"
        case 3: return "Average"
        case 4: return "Tidy"
        case 5: return "Spotless"
        default: return "Not Specified"
        }
    }
    
    private var socialLevelText: String {
        switch socialLevel {
        case 1: return "Very Private"
        case 2: return "Somewhat Private"
        case 3: return "Balanced"
        case 4: return "Social"
        case 5: return "Very Social"
        default: return "Not Specified"
        }
    }
    
    private func loadUserData() {
        // Load current user data into local state
        let user = viewModel.currentUser
        name = user.name
        email = user.email
        phone = user.phone
        age = user.age
        city = user.city
        hasPlace = user.hasPlace
        moveInDate = user.moveInDate
        budget = user.budget
        smoking = user.smoking
        pets = user.pets
        wakeUpTime = user.wakeUpTime
        sleepTime = user.sleepTime
        cleanliness = user.cleanliness
        socialLevel = user.socialLevel
        bio = user.bio
        interests = user.interests
    }
    
    private func saveChanges() {
        // Set loading state
        isSaving = true
        
        // Update the user profile with edited values
        viewModel.currentUser.name = name
        viewModel.currentUser.email = email
        viewModel.currentUser.phone = phone
        viewModel.currentUser.age = age
        viewModel.currentUser.city = city
        viewModel.currentUser.hasPlace = hasPlace
        viewModel.currentUser.moveInDate = moveInDate
        viewModel.currentUser.budget = budget
        viewModel.currentUser.smoking = smoking
        viewModel.currentUser.pets = pets
        viewModel.currentUser.wakeUpTime = wakeUpTime
        viewModel.currentUser.sleepTime = sleepTime
        viewModel.currentUser.cleanliness = cleanliness
        viewModel.currentUser.socialLevel = socialLevel
        viewModel.currentUser.bio = bio
        viewModel.currentUser.interests = interests
        
        // In a real app, you would handle image uploads to storage here
        // and update the profileImageName and propertyImageName properties
        
        // Save to Firestore
        viewModel.saveUserProfileToFirestore { error in
            isSaving = false
            
            if let error = error {
                print("Error saving profile: \(error.localizedDescription)")
                showingSaveErrorAlert = true
            } else {
                print("Profile saved successfully")
                showingSaveSuccessAlert = true
            }
        }
    }
}

#Preview {
    ProfileEditView()
        .environmentObject(RoommateViewModel())
}
