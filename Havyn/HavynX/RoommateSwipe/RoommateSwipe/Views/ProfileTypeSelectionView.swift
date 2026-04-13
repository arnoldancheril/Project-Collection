//
//  ProfileTypeSelectionView.swift
//  RoommateSwipe
//
//  Created by  on 3/25/25.
//

import SwiftUI

// Used to identify the user type during registration
enum UserType {
    case lookingForRoom
    case haveRoom
    case apartmentListing
}

struct ProfileTypeSelectionView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var selectedType: UserType?
    @State private var navigateToNextFlow = false
    
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
                    Text("Select Your Profile Type")
                        .font(.title)
                        .fontWeight(.bold)
                        .padding(.top, 40)
                    
                    Text("Choose the option that best describes your housing situation")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    
                    Spacer()
                    
                    // Profile Type Selection Cards
                    VStack(spacing: 16) {
                        ProfileTypeCard(
                            title: "Looking for a Room",
                            subtitle: "I need a place and a roommate",
                            iconName: "magnifyingglass.circle.fill",
                            isSelected: selectedType == .lookingForRoom,
                            action: { selectedType = .lookingForRoom }
                        )
                        
                        ProfileTypeCard(
                            title: "Have a Room",
                            subtitle: "I have a place and need a roommate",
                            iconName: "house.circle.fill",
                            isSelected: selectedType == .haveRoom,
                            action: { selectedType = .haveRoom }
                        )
                        
                        ProfileTypeCard(
                            title: "Apartment Listing",
                            subtitle: "I'm an apartment company listing a property",
                            iconName: "building.2.crop.circle.fill",
                            isSelected: selectedType == .apartmentListing,
                            action: { selectedType = .apartmentListing }
                        )
                    }
                    .padding(.horizontal)
                    
                    Spacer()
                    
                    // Navigation button
                    Button(action: {
                        navigateToNextFlow = true
                    }) {
                        Text("Continue")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(
                                LinearGradient(
                                    gradient: Gradient(colors: [Color.blue, Color.purple]),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                                .opacity(selectedType == nil ? 0.5 : 1)
                            )
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                    .disabled(selectedType == nil)
                    .padding()
                    
                    NavigationLink(
                        destination: destinationView,
                        isActive: $navigateToNextFlow,
                        label: { EmptyView() }
                    )
                }
            }
            .navigationBarTitle("Create Account", displayMode: .inline)
            .navigationBarItems(leading: Button("Cancel") {
                dismiss()
            })
        }
    }
    
    @ViewBuilder
    private var destinationView: some View {
        switch selectedType {
        case .lookingForRoom:
            SignUpView(onComplete: {
                dismiss()
            })
        case .haveRoom:
            HaveRoomSignUpView(onComplete: {
                dismiss()
            })
        case .apartmentListing:
            ApartmentListingSignUpView(onComplete: {
                dismiss()
            })
        case nil:
            EmptyView()
        }
    }
}

struct ProfileTypeCard: View {
    let title: String
    let subtitle: String
    let iconName: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 16) {
                // Icon
                Image(systemName: iconName)
                    .font(.system(size: 36))
                    .foregroundColor(isSelected ? .white : .blue)
                    .frame(width: 60, height: 60)
                
                // Text content
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.headline)
                        .foregroundColor(isSelected ? .white : .primary)
                    
                    Text(subtitle)
                        .font(.subheadline)
                        .foregroundColor(isSelected ? .white.opacity(0.9) : .secondary)
                        .lineLimit(2)
                }
                
                Spacer()
                
                // Selection indicator
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.white)
                        .font(.title2)
                }
            }
            .padding()
            .background(
                isSelected ? 
                LinearGradient(
                    gradient: Gradient(colors: [Color.blue, Color.purple]),
                    startPoint: .leading,
                    endPoint: .trailing
                ) : 
                LinearGradient(
                    gradient: Gradient(colors: [Color(UIColor.secondarySystemBackground), Color(UIColor.secondarySystemBackground)]),
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .cornerRadius(16)
            .shadow(color: isSelected ? Color.blue.opacity(0.3) : Color.gray.opacity(0.1), radius: 5)
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(isSelected ? Color.clear : Color.gray.opacity(0.2), lineWidth: 1)
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct ProfileTypeSelectionView_Previews: PreviewProvider {
    static var previews: some View {
        ProfileTypeSelectionView()
    }
} 