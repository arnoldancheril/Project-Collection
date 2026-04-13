//
//  ProfileDisplayView.swift
//  RoommateSwipe
//

import SwiftUI

struct ProfileDisplayView: View {
    @EnvironmentObject var viewModel: RoommateViewModel
    @Binding var isAuthenticated: Bool
    @State private var showEditView = false
    @State private var showingFirebaseTestAlert = false
    @State private var firebaseTestSuccess = false
    @State private var firebaseTestMessage = ""
    @State private var isTestingFirebase = false
    @State private var showingUploadErrorAlert = false
    @State private var showingLogoutAlert = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Profile Header with Image
                    profileHeader
                    
                    // Firebase connection test button (only visible in development)
                    #if DEBUG
                    Button(action: {
                        testFirebaseConnection()
                    }) {
                        HStack {
                            Image(systemName: "server.rack")
                            Text(isTestingFirebase ? "Testing..." : "Test Firebase Connection")
                        }
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(8)
                    }
                    .disabled(isTestingFirebase)
                    .padding(.horizontal)
                    
                    // Sample profiles upload button (only visible in development)
                    Button(action: {
                        uploadSampleProfiles()
                    }) {
                        HStack {
                            Image(systemName: "arrow.up.doc.fill")
                            if viewModel.isUploadingSampleProfiles {
                                Text("Uploading Profiles...")
                            } else if viewModel.sampleProfilesUploaded {
                                Text("Sample Profiles Uploaded")
                            } else {
                                Text("Upload Sample Profiles to Firebase")
                            }
                        }
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(viewModel.sampleProfilesUploaded ? Color.green.opacity(0.1) : Color.orange.opacity(0.1))
                        .cornerRadius(8)
                    }
                    .disabled(viewModel.isUploadingSampleProfiles || viewModel.sampleProfilesUploaded)
                    .padding(.horizontal)
                    #endif
                    
                    // Main Content Sections
                    Group {
                        // Basic Information Section
                        InfoSection(title: "Basic Information") {
                            InfoRow(icon: "person.fill", title: "Name", value: viewModel.currentUser.name)
                            InfoRow(icon: "envelope.fill", title: "Email", value: viewModel.currentUser.email)
                            InfoRow(icon: "phone.fill", title: "Phone", value: viewModel.currentUser.phone)
                            InfoRow(icon: "calendar", title: "Age", value: "\(viewModel.currentUser.age)")
                        }
                        
                        // Housing Preferences Section
                        InfoSection(title: "Housing Preferences") {
                            InfoRow(icon: "house.fill", title: "Status", value: viewModel.currentUser.hasPlace ? "Offering Room" : "Seeking Room")
                            InfoRow(icon: "mappin.circle.fill", title: "Location", value: viewModel.currentUser.city)
                            InfoRow(icon: "calendar.badge.clock", title: "Move-in", value: viewModel.currentUser.moveInDate)
                            InfoRow(icon: "dollarsign.circle.fill", title: "Budget", value: viewModel.currentUser.budget)
                        }
                        
                        // Lifestyle Section
                        InfoSection(title: "Lifestyle & Compatibility") {
                            InfoRow(icon: "smoke.fill", title: "Smoking", value: viewModel.currentUser.smoking ? "Yes" : "No")
                            InfoRow(icon: "pawprint.fill", title: "Pets", value: viewModel.currentUser.pets ? "Yes" : "No")
                            InfoRow(icon: "moon.stars.fill", title: "Schedule", value: "\(viewModel.currentUser.wakeUpTime) - \(viewModel.currentUser.sleepTime)")
                            InfoRow(icon: "sparkles", title: "Cleanliness", value: cleanlinessLevel)
                            InfoRow(icon: "person.3.fill", title: "Social", value: socialLevel)
                        }
                        
                        // Bio & Interests Section
                        InfoSection(title: "About Me") {
                            Text(viewModel.currentUser.bio)
                                .foregroundColor(.secondary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.vertical, 8)
                            
                            if !viewModel.currentUser.interests.isEmpty {
                                InterestTagsView(interests: viewModel.currentUser.interests.components(separatedBy: ", "))
                            }
                        }
                        
                        // Logout Button
                        Button(action: {
                            showingLogoutAlert = true
                        }) {
                            HStack {
                                Image(systemName: "rectangle.portrait.and.arrow.right")
                                Text("Log Out")
                            }
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.red.opacity(0.1))
                            .foregroundColor(.red)
                            .cornerRadius(8)
                        }
                        .padding(.top, 20)
                    }
                }
                .padding(.horizontal)
            }
            .navigationTitle("My Profile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showEditView = true
                    } label: {
                        Text("Edit")
                            .fontWeight(.medium)
                    }
                }
            }
            .sheet(isPresented: $showEditView) {
                ProfileEditView()
                    .environmentObject(viewModel)
            }
            .background(Color(.systemGray6))
            .alert(isPresented: $showingFirebaseTestAlert) {
                Alert(
                    title: Text(firebaseTestSuccess ? "Connection Successful" : "Connection Failed"),
                    message: Text(firebaseTestMessage),
                    dismissButton: .default(Text("OK"))
                )
            }
            .alert("Upload Error", isPresented: $showingUploadErrorAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(viewModel.uploadError ?? "Unknown error")
            }
            .alert("Log Out", isPresented: $showingLogoutAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Log Out", role: .destructive) {
                    isAuthenticated = false
                }
            } message: {
                Text("Are you sure you want to log out?")
            }
        }
    }
    
    private var profileHeader: some View {
        VStack(spacing: 16) {
            if let profileImage = viewModel.currentUser.profileImageName {
                Image(profileImage)
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
            
            Text(viewModel.currentUser.name)
                .font(.title2)
                .fontWeight(.bold)
            
            Text("\(viewModel.currentUser.age) • \(viewModel.currentUser.city)")
                .foregroundColor(.secondary)
        }
        .padding(.vertical)
    }
    
    private var cleanlinessLevel: String {
        switch viewModel.currentUser.cleanliness {
        case 1: return "Relaxed"
        case 2: return "Casual"
        case 3: return "Average"
        case 4: return "Tidy"
        case 5: return "Spotless"
        default: return "Not Specified"
        }
    }
    
    private var socialLevel: String {
        switch viewModel.currentUser.socialLevel {
        case 1: return "Very Private"
        case 2: return "Somewhat Private"
        case 3: return "Balanced"
        case 4: return "Social"
        case 5: return "Very Social"
        default: return "Not Specified"
        }
    }
    
    private func testFirebaseConnection() {
        isTestingFirebase = true
        
        viewModel.testFirebaseConnection { success, error in
            DispatchQueue.main.async {
                isTestingFirebase = false
                firebaseTestSuccess = success
                
                if success {
                    firebaseTestMessage = "Successfully connected to Firebase and wrote test data."
                } else if let error = error as? String {
                    firebaseTestMessage = "Connection failed: \(error)"
                } else {
                    firebaseTestMessage = "Connection failed for an unknown reason."
                }
                
                showingFirebaseTestAlert = true
            }
        }
    }
    
    private func uploadSampleProfiles() {
        viewModel.uploadSampleProfilesToFirebase()
        
        // Check for errors after a delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            if viewModel.uploadError != nil {
                showingUploadErrorAlert = true
            }
        }
    }
}

// MARK: - Supporting Views

struct InfoSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
                .foregroundColor(.primary)
            
            content()
                .padding()
                .background(Color.white)
                .cornerRadius(12)
                .shadow(color: Color.black.opacity(0.05), radius: 5)
        }
        .padding(.vertical, 8)
    }
}

struct InfoRow: View {
    let icon: String
    let title: String
    let value: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 24)
            
            Text(title)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Text(value)
                .foregroundColor(.primary)
        }
                    .font(.subheadline)
    }
}

struct InterestTagsView: View {
    let interests: [String]
    
    var body: some View {
        FlowLayout(alignment: .leading, spacing: 8) {
            ForEach(interests, id: \.self) { interest in
                Text(interest)
                    .font(.footnote)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.blue.opacity(0.1))
                    .foregroundColor(.blue)
                    .cornerRadius(16)
            }
        }
    }
}

struct FlowLayout: Layout {
    var alignment: HorizontalAlignment = .center
    var spacing: CGFloat = 8
    
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = computeRows(proposal: proposal, subviews: subviews)
        return computeSize(rows: rows, proposal: proposal)
    }
    
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = computeRows(proposal: proposal, subviews: subviews)
        placeRows(rows, in: bounds)
    }
    
    private func computeRows(proposal: ProposedViewSize, subviews: Subviews) -> [[LayoutSubview]] {
        var rows: [[LayoutSubview]] = [[]]
        var currentRow = 0
        var remainingWidth = proposal.width ?? 0
        
        for subview in subviews {
            let size = subview.sizeThatFits(proposal)
            if size.width > remainingWidth {
                currentRow += 1
                rows.append([])
                remainingWidth = (proposal.width ?? 0) - size.width - spacing
            } else {
                remainingWidth -= size.width + spacing
            }
            rows[currentRow].append(subview)
        }
        return rows
    }
    
    private func computeSize(rows: [[LayoutSubview]], proposal: ProposedViewSize) -> CGSize {
        var height: CGFloat = 0
        for row in rows {
            let rowHeight = row.map { $0.sizeThatFits(proposal).height }.max() ?? 0
            height += rowHeight + spacing
        }
        return CGSize(width: proposal.width ?? 0, height: height)
    }
    
    private func placeRows(_ rows: [[LayoutSubview]], in bounds: CGRect) {
        var y = bounds.minY
        for row in rows {
            var x = bounds.minX
            let rowHeight = row.map { $0.sizeThatFits(.unspecified).height }.max() ?? 0
            for subview in row {
                let size = subview.sizeThatFits(.unspecified)
                subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += rowHeight + spacing
        }
    }
}

#Preview {
    ProfileDisplayView(isAuthenticated: .constant(true))
        .environmentObject(RoommateViewModel())
}
