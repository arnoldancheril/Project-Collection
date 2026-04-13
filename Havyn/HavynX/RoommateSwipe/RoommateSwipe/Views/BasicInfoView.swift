//
//  BasicInfoView.swift
//  RoommateSwipe
//

import SwiftUI

struct BasicInfoView: View {
    @Binding var userData: UserRegistrationData
    @State private var confirmPassword = ""
    
    var body: some View {
        VStack(spacing: 25) {
            Text("Tell us about yourself")
                .font(.title2)
                .fontWeight(.bold)
            
            VStack(alignment: .leading, spacing: 20) {
                // Name fields
                HStack(spacing: 15) {
                    VStack(alignment: .leading) {
                        Text("First Name")
                            .foregroundColor(.gray)
                        TextField("John", text: $userData.firstName)
                            .textFieldStyle(.roundedBorder)
                    }
                    
                    VStack(alignment: .leading) {
                        Text("Last Name")
                            .foregroundColor(.gray)
                        TextField("Doe", text: $userData.lastName)
                            .textFieldStyle(.roundedBorder)
                    }
                }
                
                // Email
                VStack(alignment: .leading) {
                    Text("Email")
                        .foregroundColor(.gray)
                    TextField("your@email.com", text: $userData.email)
                        .textFieldStyle(.roundedBorder)
                        .keyboardType(.emailAddress)
                        .autocapitalization(.none)
                }
                
                // Password fields
                VStack(alignment: .leading) {
                    Text("Password")
                        .foregroundColor(.gray)
                    SecureField("Create password", text: $userData.password)
                        .textFieldStyle(.roundedBorder)
                }
                
                VStack(alignment: .leading) {
                    Text("Confirm Password")
                        .foregroundColor(.gray)
                    SecureField("Confirm password", text: $confirmPassword)
                        .textFieldStyle(.roundedBorder)
                }
                
                // Date of Birth
                VStack(alignment: .leading) {
                    Text("Date of Birth")
                        .foregroundColor(.gray)
                    DatePicker("", selection: $userData.dateOfBirth, 
                             displayedComponents: .date)
                        .datePickerStyle(.compact)
                }
                
                // Gender Selection
                VStack(alignment: .leading) {
                    Text("Gender")
                        .foregroundColor(.gray)
                    Picker("Gender", selection: $userData.gender) {
                        Text("Male").tag("Male")
                        Text("Female").tag("Female")
                        Text("Non-binary").tag("Non-binary")
                        Text("Prefer not to say").tag("Prefer not to say")
                    }
                    .pickerStyle(.segmented)
                }
            }
            .padding(.horizontal)
        }
    }
}

#Preview {
    BasicInfoView(userData: .constant(UserRegistrationData()))
        .padding()
} 