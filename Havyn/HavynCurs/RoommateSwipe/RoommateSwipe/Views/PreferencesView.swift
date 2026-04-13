//
//  PreferencesView.swift
//  RoommateSwipe
//

import SwiftUI

struct PreferencesView: View {
    @Binding var userData: UserRegistrationData
    
    var body: some View {
        VStack(spacing: 25) {
            Text("Roommate Preferences")
                .font(.title2)
                .fontWeight(.bold)
            
            ScrollView {
                VStack(alignment: .leading, spacing: 25) {
                    // Preferred Gender
                    VStack(alignment: .leading) {
                        Text("Preferred Roommate Gender")
                            .fontWeight(.medium)
                        
                        Picker("Gender Preference", selection: $userData.preferredGender) {
                            Text("Any").tag("Any")
                            Text("Male").tag("Male")
                            Text("Female").tag("Female")
                            Text("Non-binary").tag("Non-binary")
                        }
                        .pickerStyle(.segmented)
                    }
                    
                    // Age Range
                    VStack(alignment: .leading) {
                        Text("Preferred Age Range")
                            .fontWeight(.medium)
                        
                        HStack {
                            Text("\(Int(userData.ageRange.lowerBound))")
                            Slider(
                                value: .init(
                                    get: { Double(userData.ageRange.lowerBound) },
                                    set: { userData.ageRange = Int($0)...userData.ageRange.upperBound }
                                ),
                                in: 18...50
                            )
                            Text("\(Int(userData.ageRange.upperBound))")
                            Slider(
                                value: .init(
                                    get: { Double(userData.ageRange.upperBound) },
                                    set: { userData.ageRange = userData.ageRange.lowerBound...Int($0) }
                                ),
                                in: 18...50
                            )
                        }
                    }
                    
                    // Rent Range
                    VStack(alignment: .leading) {
                        Text("Monthly Rent Budget")
                            .fontWeight(.medium)
                        
                        HStack {
                            Text("$\(Int(userData.rentRange.lowerBound))")
                            Slider(
                                value: .init(
                                    get: { Double(userData.rentRange.lowerBound) },
                                    set: { userData.rentRange = Int($0)...userData.rentRange.upperBound }
                                ),
                                in: 500...5000,
                                step: 100
                            )
                            Text("$\(Int(userData.rentRange.upperBound))")
                            Slider(
                                value: .init(
                                    get: { Double(userData.rentRange.upperBound) },
                                    set: { userData.rentRange = userData.rentRange.lowerBound...Int($0) }
                                ),
                                in: 500...5000,
                                step: 100
                            )
                        }
                    }
                    
                    // Move-in Date
                    VStack(alignment: .leading) {
                        Text("Preferred Move-in Date")
                            .fontWeight(.medium)
                        
                        DatePicker(
                            "Move-in Date",
                            selection: $userData.moveInDate,
                            displayedComponents: [.date]
                        )
                        .datePickerStyle(.graphical)
                    }
                    
                    // Bio
                    VStack(alignment: .leading) {
                        Text("Tell potential roommates about yourself")
                            .fontWeight(.medium)
                        
                        TextEditor(text: $userData.bio)
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
    PreferencesView(userData: .constant(UserRegistrationData()))
        .padding()
} 