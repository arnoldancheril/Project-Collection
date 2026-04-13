//
//  LoginView.swift
//  RoommateSwipe
//

import SwiftUI

struct LoginView: View {
    @Binding var isAuthenticated: Bool
    @Binding var isApartmentLister: Bool
    @State private var showingSignUp = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Havyn")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .foregroundColor(.blue)
                
                Button(action: {
                    isAuthenticated = true
                }) {
                    Text("Log In")
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(10)
                }
                .padding(.horizontal, 30)
                
                Button(action: {
                    showingSignUp = true
                }) {
                    Text("Create Account")
                        .foregroundColor(.blue)
                }
                
                Divider()
                    .padding(.vertical, 10)
                
                Button(action: {
                    isApartmentLister = true
                }) {
                    Text("List an Apartment")
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .cornerRadius(10)
                }
                .padding(.horizontal, 30)
            }
            .padding()
            .sheet(isPresented: $showingSignUp) {
                ProfileTypeSelectionView()
            }
        }
    }
}

#Preview {
    LoginView(isAuthenticated: .constant(false), isApartmentLister: .constant(false))
} 