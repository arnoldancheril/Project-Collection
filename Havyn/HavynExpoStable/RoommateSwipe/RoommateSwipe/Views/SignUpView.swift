//
//  SignUpView.swift
//  RoommateSwipe
//

import SwiftUI

struct SignUpView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var currentStep = 0
    @State private var userData = UserRegistrationData()
    @State private var isLoading = false
    @State private var showAlert = false
    @State private var alertTitle = ""
    @State private var alertMessage = ""
    var onComplete: (() -> Void)?
    
    var body: some View {
        NavigationView {
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
                    ProgressBar(currentStep: currentStep, totalSteps: 6)
                        .padding(.horizontal)
                    
                    // Step content
                    switch currentStep {
                    case 0:
                        BasicInfoView(userData: $userData)
                            .transition(.slide)
                    case 1:
                        LifestyleQuestionsView(userData: $userData)
                            .transition(.slide)
                    case 2:
                        PreferencesView(userData: $userData)
                            .transition(.slide)
                    case 3:
                        ScheduleView(userData: $userData)
                            .transition(.slide)
                    case 4:
                        PropertyPreferencesView(userData: $userData)
                            .transition(.slide)
                    case 5:
                        PhotoUploadView(userData: $userData)
                            .transition(.slide)
                    default:
                        EmptyView()
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
                                if currentStep < 5 {
                                    currentStep += 1
                                } else {
                                    // Registration complete - save to Firebase
                                    saveToFirebase()
                                }
                            }
                        }) {
                            Text(currentStep == 5 ? "Complete" : "Next")
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
            .navigationBarTitle("Create Your Profile", displayMode: .inline)
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
    }
    
    private func saveToFirebase() {
        isLoading = true
        
        // Create a unique user ID 
        let userId = UUID().uuidString
        
        FirebaseService.shared.saveLookingForRoomUser(userData, userId: userId) { userId, error in
            isLoading = false
            
            if let error = error {
                alertTitle = "Error"
                alertMessage = "Failed to save your profile: \(error.localizedDescription)"
                showAlert = true
            } else {
                alertTitle = "Success"
                alertMessage = "Your profile has been created successfully!"
                showAlert = true
                
                // If we had image storage set up, we would upload images here
                // For now, we're just recording that they exist
            }
        }
    }
}

struct ProgressBar: View {
    let currentStep: Int
    let totalSteps: Int
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Rectangle()
                    .foregroundColor(Color.gray.opacity(0.2))
                    .frame(height: 8)
                    .cornerRadius(4)
                
                Rectangle()
                    .foregroundColor(.blue)
                    .frame(width: geometry.size.width * CGFloat(Double(currentStep + 1) / Double(totalSteps)), height: 8)
                    .cornerRadius(4)
            }
        }
        .frame(height: 8)
    }
}

#Preview {
    SignUpView()
} 