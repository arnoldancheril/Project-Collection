//
//  ProfileView.swift
//  RoommateSwipe
//
//  Created by AA on 2/27/25.
//

import SwiftUI

struct ProfileView: View {
    @EnvironmentObject var viewModel: RoommateViewModel

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Basic Info")) {
                    TextField("Name", text: $viewModel.currentUser.name)
                    TextField("Age", value: $viewModel.currentUser.age, format: .number)
                        .keyboardType(.numberPad)
                    TextField("City", text: $viewModel.currentUser.city)
                }

                Section(header: Text("Roommate Preferences")) {
                    TextField("Budget", text: $viewModel.currentUser.budget)
                    TextField("Interests/Hobbies", text: $viewModel.currentUser.interests)
                    TextField("Bio", text: $viewModel.currentUser.bio, axis: .vertical)
                }
                
                Section {
                    Button("Save Profile") {
                        // In a real app, you'd persist this to a backend or local storage
                        print("Profile saved: \(viewModel.currentUser)")
                    }
                }
            }
            .navigationTitle("My Profile")
        }
    }
}

#Preview {
    ProfileView()
        .environmentObject(RoommateViewModel())
}
