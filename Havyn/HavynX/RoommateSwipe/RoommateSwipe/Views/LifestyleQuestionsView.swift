//
//  LifestyleQuestionsView.swift
//  RoommateSwipe
//

import SwiftUI

struct LifestyleQuestionsView: View {
    @Binding var userData: UserRegistrationData
    
    var body: some View {
        VStack(spacing: 25) {
            Text("Your Lifestyle")
                .font(.title2)
                .fontWeight(.bold)
            
            ScrollView {
                VStack(alignment: .leading, spacing: 25) {
                    // Cleanliness Level
                    VStack(alignment: .leading) {
                        Text("How clean do you keep your space?")
                            .fontWeight(.medium)
                        
                        Slider(value: .init(get: {
                            Double(userData.cleanliness)
                        }, set: { newValue in
                            userData.cleanliness = Int(newValue)
                        }), in: 1...5, step: 1)
                        
                        HStack {
                            Text("Casual")
                            Spacer()
                            Text("Average")
                            Spacer()
                            Text("Spotless")
                        }
                        .font(.caption)
                        .foregroundColor(.gray)
                    }
                    
                    // Noise Level
                    VStack(alignment: .leading) {
                        Text("What's your typical noise level?")
                            .fontWeight(.medium)
                        
                        Slider(value: .init(get: {
                            Double(userData.noise)
                        }, set: { newValue in
                            userData.noise = Int(newValue)
                        }), in: 1...5, step: 1)
                        
                        HStack {
                            Text("Silent")
                            Spacer()
                            Text("Moderate")
                            Spacer()
                            Text("Lively")
                        }
                        .font(.caption)
                        .foregroundColor(.gray)
                    }
                    
                    // Social Level
                    VStack(alignment: .leading) {
                        Text("How social are you at home?")
                            .fontWeight(.medium)
                        
                        Slider(value: .init(get: {
                            Double(userData.socialLevel)
                        }, set: { newValue in
                            userData.socialLevel = Int(newValue)
                        }), in: 1...5, step: 1)
                        
                        HStack {
                            Text("Private")
                            Spacer()
                            Text("Balanced")
                            Spacer()
                            Text("Very Social")
                        }
                        .font(.caption)
                        .foregroundColor(.gray)
                    }
                    
                    // Sleep Schedule
                    VStack(alignment: .leading) {
                        Text("What's your sleep schedule like?")
                            .fontWeight(.medium)
                        
                        Picker("Sleep Schedule", selection: $userData.sleepSchedule) {
                            Text("Early Bird").tag("Early Bird")
                            Text("Night Owl").tag("Night Owl")
                            Text("Regular Schedule").tag("Regular Schedule")
                            Text("Variable").tag("Variable")
                        }
                        .pickerStyle(.segmented)
                    }
                    
                    // Habits
                    VStack(alignment: .leading, spacing: 15) {
                        Text("Habits & Preferences")
                            .fontWeight(.medium)
                        
                        Toggle("Do you drink alcohol?", isOn: $userData.drinking)
                        Toggle("Do you smoke?", isOn: $userData.smoking)
                        Toggle("Do you have or plan to have pets?", isOn: $userData.pets)
                    }
                }
                .padding()
            }
        }
    }
}

#Preview {
    LifestyleQuestionsView(userData: .constant(UserRegistrationData()))
        .padding()
} 