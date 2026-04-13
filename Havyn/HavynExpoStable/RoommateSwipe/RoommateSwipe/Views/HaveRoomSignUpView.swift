//
//  HaveRoomSignUpView.swift
//  RoommateSwipe
//
//  Created by  on 3/25/25.
//

import SwiftUI

struct HaveRoomSignUpView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var currentStep = 0
    @State private var userData = HaveRoomRegistrationData()
    @State private var isLoading = false
    @State private var showAlert = false
    @State private var alertTitle = ""
    @State private var alertMessage = ""
    var onComplete: (() -> Void)?
    
    var body: some View {
        ZStack {
            // Background gradient
            LinearGradient(
                gradient: Gradient(colors: [Color.blue.opacity(0.1), Color.purple.opacity(0.1)]),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            VStack(spacing: 20) {
                // Progress bar
                ProgressBar(currentStep: currentStep, totalSteps: 4)
                    .padding(.horizontal)
                
                // Step title
                Text(stepTitle)
                    .font(.title2)
                    .fontWeight(.bold)
                    .padding(.top, 10)
                
                // Step content
                ScrollView {
                    VStack(spacing: 20) {
                        switch currentStep {
                        case 0:
                            HaveRoomBasicInfoView(userData: $userData)
                                .transition(.opacity)
                        case 1:
                            AboutSpaceView(userData: $userData)
                                .transition(.opacity)
                        case 2:
                            RoommatePreferencesView(userData: $userData)
                                .transition(.opacity)
                        case 3:
                            HaveRoomPersonalizationView(userData: $userData)
                                .transition(.opacity)
                        default:
                            EmptyView()
                        }
                    }
                    .padding()
                }
                
                Spacer()
                
                // Navigation buttons
                HStack(spacing: 20) {
                    if currentStep > 0 {
                        Button(action: {
                            withAnimation {
                                currentStep -= 1
                            }
                        }) {
                            Text("Back")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.gray.opacity(0.2))
                                .cornerRadius(12)
                        }
                    }
                    
                    Button(action: {
                        withAnimation {
                            if currentStep < 3 {
                                currentStep += 1
                            } else {
                                // Registration complete - save to Firebase
                                saveToFirebase()
                            }
                        }
                    }) {
                        Text(currentStep == 3 ? "Complete" : "Next")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(
                                LinearGradient(
                                    gradient: Gradient(colors: [Color.blue, Color.purple]),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                    .disabled(isLoading)
                }
                .padding()
            }
            
            // Loading overlay
            if isLoading {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()
                
                VStack {
                    ProgressView()
                        .scaleEffect(1.5)
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .padding()
                    
                    Text("Saving your profile...")
                        .foregroundColor(.white)
                        .font(.headline)
                }
                .padding(30)
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.gray.opacity(0.7))
                )
            }
        }
        .navigationBarTitle("Have a Room", displayMode: .inline)
        .navigationBarBackButtonHidden(true)
        .navigationBarItems(leading: Button("Cancel") {
            dismiss()
        })
        .alert(isPresented: $showAlert) {
            Alert(
                title: Text(alertTitle),
                message: Text(alertMessage),
                dismissButton: .default(Text("OK")) {
                    if alertTitle == "Success" {
                        onComplete?()
                    }
                }
            )
        }
    }
    
    private var stepTitle: String {
        switch currentStep {
        case 0:
            return "Basic Information"
        case 1:
            return "About Your Space"
        case 2:
            return "Roommate Preferences"
        case 3:
            return "Personalization & Interests"
        default:
            return ""
        }
    }
    
    private func saveToFirebase() {
        isLoading = true
        
        // Create a unique user ID
        let userId = UUID().uuidString
        
        FirebaseService.shared.saveHaveRoomUser(userData, userId: userId) { userId, error in
            isLoading = false
            
            if let error = error {
                alertTitle = "Error"
                alertMessage = "Failed to save your profile: \(error.localizedDescription)"
                showAlert = true
            } else {
                alertTitle = "Success"
                alertMessage = "Your 'Have a Room' profile has been created successfully!"
                showAlert = true
                
                // If we had image storage set up, we would upload images here
                // For now, we're just recording that they exist
            }
        }
    }
}

// Step 1: Basic Information
struct HaveRoomBasicInfoView: View {
    @Binding var userData: HaveRoomRegistrationData
    
    var body: some View {
        VStack(spacing: 20) {
            FormField(title: "Full Name", text: $userData.fullName, placeholder: "Enter your full name")
            
            FormField(title: "Email Address", text: $userData.email, placeholder: "Enter your email address", keyboardType: .emailAddress)
            
            SecureFormField(title: "Password", text: $userData.password, placeholder: "Create a password")
            
            FormField(title: "Phone Number (Optional)", text: $userData.phoneNumber, placeholder: "Enter your phone number", keyboardType: .phonePad)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Date of Birth")
                    .font(.headline)
                
                DatePicker("", selection: $userData.dateOfBirth, displayedComponents: .date)
                    .datePickerStyle(WheelDatePickerStyle())
                    .labelsHidden()
                    .frame(maxHeight: 180)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
            }
        }
    }
}

// Step 2: About Your Space
struct AboutSpaceView: View {
    @Binding var userData: HaveRoomRegistrationData
    
    private let housingTypes = ["Apartment", "House", "Shared Space"]
    
    var body: some View {
        VStack(spacing: 20) {
            FormField(title: "City", text: $userData.city, placeholder: "Enter city")
            
            FormField(title: "Neighborhood", text: $userData.neighborhood, placeholder: "Enter neighborhood")
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Available Move-in Date")
                    .font(.headline)
                
                DatePicker("", selection: $userData.moveInDate, displayedComponents: .date)
                    .datePickerStyle(WheelDatePickerStyle())
                    .labelsHidden()
                    .frame(maxHeight: 180)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Rent Price ($)")
                    .font(.headline)
                
                HStack {
                    Text("$\(userData.rentPrice)")
                        .frame(width: 70, alignment: .leading)
                    
                    Slider(value: Binding(
                        get: { Double(userData.rentPrice) },
                        set: { userData.rentPrice = Int($0) }
                    ), in: 300...5000, step: 50)
                }
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Type of Housing")
                    .font(.headline)
                
                Picker("", selection: $userData.housingType) {
                    ForEach(housingTypes, id: \.self) { type in
                        Text(type).tag(type)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Room Furnishing")
                    .font(.headline)
                
                Toggle("Furnished", isOn: $userData.isFurnished)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
            }
        }
    }
}

// Step 3: Roommate Preferences
struct RoommatePreferencesView: View {
    @Binding var userData: HaveRoomRegistrationData
    
    private let genderOptions = ["Male", "Female", "Non-binary", "No Preference"]
    private let lifestyleOptions = ["Morning person", "Night owl", "Flexible"]
    
    var body: some View {
        VStack(spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Gender Preference")
                    .font(.headline)
                
                Picker("", selection: $userData.genderPreference) {
                    ForEach(genderOptions, id: \.self) { option in
                        Text(option).tag(option)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Smoking Preferences")
                    .font(.headline)
                
                Toggle("Allow smoking", isOn: $userData.smokingPreference)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Pet Preferences")
                    .font(.headline)
                
                Toggle("Allow pets", isOn: $userData.petPreference)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Lifestyle Match")
                    .font(.headline)
                
                Picker("", selection: $userData.lifestylePreference) {
                    ForEach(lifestyleOptions, id: \.self) { option in
                        Text(option).tag(option)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Cleanliness Level")
                    .font(.headline)
                
                HStack {
                    Text("Relaxed")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Slider(value: Binding(
                        get: { Double(userData.cleanlinessLevel) },
                        set: { userData.cleanlinessLevel = Int($0) }
                    ), in: 1...5, step: 1)
                    
                    Text("Neat")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
            }
        }
    }
}

// Step 4: Personalization and Interests
struct HaveRoomPersonalizationView: View {
    @Binding var userData: HaveRoomRegistrationData
    
    private let sampleInterests = [
        "Reading", "Gaming", "Cooking", "Fitness", "Music", "Movies", "Travel", 
        "Art", "Photography", "Hiking", "Yoga", "Technology", "Sports", "Dancing",
        "Meditation", "Writing", "Fashion", "DIY", "Gardening", "Pets"
    ]
    
    var body: some View {
        VStack(spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Short Bio")
                    .font(.headline)
                
                TextEditor(text: $userData.bio)
                    .frame(height: 120)
                    .padding(4)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                    )
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Interests and Hobbies")
                    .font(.headline)
                
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))], spacing: 10) {
                        ForEach(sampleInterests, id: \.self) { interest in
                            InterestTag(
                                interest: interest,
                                isSelected: userData.interests.contains(interest),
                                action: {
                                    if userData.interests.contains(interest) {
                                        userData.interests.removeAll { $0 == interest }
                                    } else {
                                        userData.interests.append(interest)
                                    }
                                }
                            )
                        }
                    }
                }
                .frame(height: 200)
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Upload Profile Picture")
                    .font(.headline)
                
                Button(action: {
                    // Photo picker would go here - not implementing for this mockup
                }) {
                    HStack {
                        Image(systemName: "camera")
                        Text("Choose Photo")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .foregroundColor(.blue)
                    .cornerRadius(10)
                }
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Upload Room Photos")
                    .font(.headline)
                
                Button(action: {
                    // Photo picker would go here - not implementing for this mockup
                }) {
                    HStack {
                        Image(systemName: "photo.on.rectangle")
                        Text("Choose Photos")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .foregroundColor(.blue)
                    .cornerRadius(10)
                }
            }
        }
    }
}

struct InterestTag: View {
    let interest: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(interest)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(isSelected ? Color.blue : Color.gray.opacity(0.2))
                .foregroundColor(isSelected ? .white : .primary)
                .cornerRadius(16)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct HaveRoomSignUpView_Previews: PreviewProvider {
    static var previews: some View {
        HaveRoomSignUpView()
    }
} 