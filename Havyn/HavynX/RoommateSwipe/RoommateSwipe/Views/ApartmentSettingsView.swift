//
//  ApartmentSettingsView.swift
//  RoommateSwipe
//

import SwiftUI

struct ApartmentSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var companyName = "Westside Property Management"
    @State private var contactName = "John Smith"
    @State private var contactEmail = "john@westsidepm.com"
    @State private var contactPhone = "(555) 123-4567"
    @State private var businessAddress = "123 Main Street, Suite 400, San Francisco, CA 94105"
    
    @State private var notificationsEnabled = true
    @State private var instantMessagingEnabled = true
    @State private var emailNotificationsEnabled = true
    @State private var autoSchedulingEnabled = false
    
    @State private var showingLogoutAlert = false
    @State private var showingDeleteAccountAlert = false
    
    var body: some View {
        NavigationView {
            List {
                // Company Information Section
                Section(header: Text("COMPANY INFORMATION")) {
                    VStack(alignment: .center, spacing: 16) {
                        Image(systemName: "building.2")
                            .font(.system(size: 60))
                            .foregroundColor(.blue)
                            .padding()
                        
                        Text(companyName)
                            .font(.headline)
                        
                        Text("Property Management")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        Button(action: {
                            // Action to edit profile
                        }) {
                            Text("Edit Company Profile")
                                .font(.footnote)
                                .foregroundColor(.blue)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Company Name", value: companyName)
                    }
                    
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Contact Person", value: contactName)
                    }
                    
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Email", value: contactEmail)
                    }
                    
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Phone", value: contactPhone)
                    }
                    
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Business Address", value: businessAddress)
                    }
                }
                
                // Notification Settings
                Section(header: Text("NOTIFICATIONS")) {
                    Toggle("Enable Notifications", isOn: $notificationsEnabled)
                    
                    if notificationsEnabled {
                        Toggle("Instant Messages", isOn: $instantMessagingEnabled)
                        Toggle("Email Notifications", isOn: $emailNotificationsEnabled)
                        Toggle("Auto-Scheduling", isOn: $autoSchedulingEnabled)
                    }
                }
                
                // App Settings
                Section(header: Text("APP SETTINGS")) {
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Language", value: "English")
                    }
                    
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Location Services", value: "Enabled")
                    }
                    
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Change Password", value: "")
                    }
                }
                
                // Support
                Section(header: Text("SUPPORT")) {
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Help Center", value: "")
                    }
                    
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Contact Support", value: "")
                    }
                    
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Terms of Service", value: "")
                    }
                    
                    NavigationLink(destination: EmptyView()) {
                        SettingsRow(title: "Privacy Policy", value: "")
                    }
                }
                
                // Actions
                Section {
                    Button(action: {
                        showingLogoutAlert = true
                    }) {
                        HStack {
                            Text("Log Out")
                                .foregroundColor(.red)
                            Spacer()
                        }
                    }
                    .alert("Log Out", isPresented: $showingLogoutAlert) {
                        Button("Cancel", role: .cancel) { }
                        Button("Log Out", role: .destructive) {
                            dismiss()
                        }
                    } message: {
                        Text("Are you sure you want to log out?")
                    }
                    
                    Button(action: {
                        showingDeleteAccountAlert = true
                    }) {
                        HStack {
                            Text("Delete Account")
                                .foregroundColor(.red)
                            Spacer()
                        }
                    }
                    .alert("Delete Account", isPresented: $showingDeleteAccountAlert) {
                        Button("Cancel", role: .cancel) { }
                        Button("Delete", role: .destructive) {
                            // Delete account action would go here
                            dismiss()
                        }
                    } message: {
                        Text("This action cannot be undone. All your data, listings, and messages will be permanently deleted.")
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Settings")
        }
    }
}

struct SettingsRow: View {
    let title: String
    let value: String
    
    var body: some View {
        HStack {
            Text(title)
                .foregroundColor(.primary)
            
            Spacer()
            
            Text(value)
                .foregroundColor(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }
}

#Preview {
    ApartmentSettingsView()
} 