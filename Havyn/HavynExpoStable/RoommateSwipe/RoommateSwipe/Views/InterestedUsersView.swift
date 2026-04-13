//
//  InterestedUsersView.swift
//  RoommateSwipe
//

import SwiftUI

struct InterestedUsersView: View {
    @State private var interestedUsers: [InterestedUser] = []
    @State private var selectedListing: ApartmentListing?
    @State private var listings: [ApartmentListing] = []
    @State private var selectedUser: InterestedUser?
    @State private var showUserProfile = false
    @State private var showingChatSheet = false
    
    var body: some View {
        NavigationView {
            VStack {
                if listings.isEmpty {
                    emptyStateView
                } else {
                    VStack {
                        // Listing selector
                        Picker("Select Listing", selection: $selectedListing) {
                            Text("All Listings").tag(nil as ApartmentListing?)
                            ForEach(listings) { listing in
                                Text(listing.name).tag(listing as ApartmentListing?)
                            }
                        }
                        .pickerStyle(.menu)
                        .padding(.horizontal)
                        
                        // Filter tabs
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 16) {
                                ForEach(InterestStatus.allCases, id: \.self) { status in
                                    FilterTab(title: status.rawValue, count: countForStatus(status), isSelected: true)
                                        .frame(minWidth: 60)
                                }
                            }
                            .padding(.horizontal)
                        }
                        .padding(.vertical, 4)
                        
                        // Divider
                        Rectangle()
                            .fill(Color(.systemGray5))
                            .frame(height: 1)
                        
                        if interestedUsers.isEmpty {
                            VStack(spacing: 12) {
                                Image(systemName: "person.fill.questionmark")
                                    .font(.system(size: 40))
                                    .foregroundColor(.gray)
                                
                                Text("No interested users yet")
                                    .font(.headline)
                                
                                Text("When users express interest in your listing, they'll appear here")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal)
                            }
                            .padding()
                            .frame(maxHeight: .infinity)
                        } else {
                            List {
                                ForEach(interestedUsers) { user in
                                    InterestedUserRow(user: user)
                                        .onTapGesture {
                                            selectedUser = user
                                            showUserProfile = true
                                        }
                                        .swipeActions {
                                            Button {
                                                selectedUser = user
                                                showingChatSheet = true
                                            } label: {
                                                Label("Message", systemImage: "message")
                                            }
                                            .tint(.blue)
                                            
                                            Button(role: .destructive) {
                                                // Reject user
                                            } label: {
                                                Label("Reject", systemImage: "xmark.circle")
                                            }
                                        }
                                }
                            }
                            .listStyle(.plain)
                        }
                    }
                }
            }
            .navigationTitle("Interested Users")
            .onAppear {
                // Load data - for testing, we'll use the sample listings and create mock interested users
                listings = ApartmentListing.sampleListings()
                if !listings.isEmpty {
                    selectedListing = nil // Show for all listings initially
                    loadInterestedUsers()
                }
            }
            .sheet(isPresented: $showUserProfile) {
                if let user = selectedUser {
                    // In a real implementation, this would show a more detailed profile
                    UserProfileSheet(user: user)
                }
            }
            .sheet(isPresented: $showingChatSheet) {
                if let user = selectedUser {
                    // In a real implementation, this would show a chat interface
                    ApartmentChatView(user: user)
                }
            }
        }
    }
    
    private var emptyStateView: some View {
        VStack(spacing: 20) {
            Image(systemName: "person.3.sequence")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text("No Listings Yet")
                .font(.title2)
                .foregroundColor(.primary)
            
            Text("Add property listings to see interested users here")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            
            NavigationLink(destination: ListingsView()) {
                HStack {
                    Image(systemName: "plus.circle.fill")
                    Text("Add Listing")
                }
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .padding(.top, 10)
        }
        .padding()
    }
    
    // Helper function to count users by status
    private func countForStatus(_ status: InterestStatus) -> Int {
        let filteredUsers = interestedUsers.filter { user in
            if let selected = selectedListing {
                return user.status == status && user.listingId == selected.id
            } else {
                return user.status == status
            }
        }
        return filteredUsers.count
    }
    
    // Load interested users based on the selected listing
    private func loadInterestedUsers() {
        // This would typically fetch from a database
        // For now, we'll create sample data
        interestedUsers = createSampleInterestedUsers()
    }
    
    // Mock data for testing
    private func createSampleInterestedUsers() -> [InterestedUser] {
        guard !listings.isEmpty else { return [] }
        
        let names = ["John Smith", "Emily Johnson", "Michael Davis", "Sarah Wilson", "David Thompson"]
        let occupations = ["Software Developer", "Marketing Manager", "Teacher", "Nurse", "Financial Analyst"]
        let messages = [
            "I'm very interested in this apartment! It's exactly what I'm looking for.",
            "Hi there! This apartment looks amazing. When can I schedule a viewing?",
            "I love the neighborhood! Is it possible to get more information about the utilities?",
            "The place looks great in the photos. Is public transportation nearby?",
            "I'm relocating for work and this would be perfect. Is it available for the dates listed?"
        ]
        
        var users: [InterestedUser] = []
        
        // Create 5-10 random interested users across all listings
        for i in 0..<min(15, names.count * listings.count) {
            let listingIndex = i % listings.count
            let nameIndex = i % names.count
            let listing = listings[listingIndex]
            
            let user = InterestedUser(
                id: "user\(i+1)",
                name: names[nameIndex],
                age: Int.random(in: 22...45),
                occupation: occupations[nameIndex],
                budget: Double.random(in: listing.monthlyRent * 0.8...listing.monthlyRent * 1.2),
                moveInDate: Date().addingTimeInterval(Double.random(in: 86400 * 7...86400 * 60)), // 1-8 weeks from now
                profileImage: "profile\(nameIndex+1)",
                initialMessage: messages[nameIndex],
                listingId: listing.id,
                dateInterested: Date().addingTimeInterval(-Double.random(in: 0...86400 * 14)), // up to 2 weeks ago
                messages: [],
                status: InterestStatus.allCases.randomElement() ?? .new
            )
            
            users.append(user)
        }
        
        return users
    }
}

struct InterestedUserRow: View {
    let user: InterestedUser
    
    var body: some View {
        HStack(spacing: 12) {
            // Profile image
            Image(user.profileImage ?? "defaultProfile")
                .resizable()
                .scaledToFill()
                .frame(width: 50, height: 50)
                .clipShape(Circle())
            
            // User info
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(user.name)
                        .font(.headline)
                    
                    Text("\(user.age)")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Text(user.occupation)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                HStack {
                    Text("Budget: $\(Int(user.budget))/mo")
                        .font(.caption)
                        .foregroundColor(.blue)
                    
                    Spacer()
                    
                    // Time since interested
                    Text(timeAgoString(from: user.dateInterested))
                        .font(.caption)
                        .foregroundColor(.gray)
                }
            }
            
            Spacer()
            
            // Status indicator
            StatusBadge(status: user.status)
        }
        .padding(.vertical, 6)
    }
    
    private func timeAgoString(from date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}

struct StatusBadge: View {
    let status: InterestStatus
    
    var body: some View {
        Text(status.rawValue)
            .font(.caption2)
            .fontWeight(.medium)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(status.color.opacity(0.2))
            .foregroundColor(status.color)
            .cornerRadius(6)
    }
}

struct FilterTab: View {
    let title: String
    let count: Int
    let isSelected: Bool
    
    // Get a shorter display name for the tab
    private var shortTitle: String {
        switch title {
        case "Tour Scheduled": return "Tour"
        case "Applied": return "Applied"
        case "Approved": return "Approved"
        case "Rejected": return "Rejected"
        case "Withdrawn": return "Withdrawn"
        case "Contacted": return "Contacted"
        case "New": return "New"
        default: return title
        }
    }
    
    var body: some View {
        VStack(spacing: 6) {
            Text(shortTitle)
                .font(.caption)
                .fontWeight(isSelected ? .semibold : .regular)
                .lineLimit(1)
            
            Text("\(count)")
                .font(.caption2)
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
                .background(isSelected ? Color.blue : Color.gray.opacity(0.2))
                .foregroundColor(isSelected ? .white : .secondary)
                .cornerRadius(10)
            
            if isSelected {
                Rectangle()
                    .fill(Color.blue)
                    .frame(height: 2)
            } else {
                Rectangle()
                    .fill(Color.clear)
                    .frame(height: 2)
            }
        }
        .frame(maxWidth: .infinity)
    }
}

// Mock views for previews
struct UserProfileSheet: View {
    let user: InterestedUser
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Image(user.profileImage ?? "defaultProfile")
                        .resizable()
                        .scaledToFill()
                        .frame(height: 180)
                        .frame(maxWidth: .infinity)
                        .clipped()
                    
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(user.name)
                                .font(.title2)
                                .fontWeight(.bold)
                            
                            Text("\(user.age)")
                                .font(.title3)
                                .foregroundColor(.secondary)
                        }
                        
                        Text(user.occupation)
                            .font(.headline)
                            .foregroundColor(.secondary)
                        
                        // Additional profile details would go here
                        
                        Text("Message from \(user.name):")
                            .font(.headline)
                            .padding(.top, 8)
                        
                        Text(user.initialMessage)
                            .font(.body)
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(8)
                        
                        HStack(spacing: 12) {
                            Button(action: {}) {
                                Text("Schedule")
                                    .fontWeight(.medium)
                                    .padding()
                                    .frame(maxWidth: .infinity)
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                            
                            Button(action: {}) {
                                Text("Message")
                                    .fontWeight(.medium)
                                    .padding()
                                    .frame(maxWidth: .infinity)
                                    .background(Color(.systemGray5))
                                    .foregroundColor(.primary)
                                    .cornerRadius(10)
                            }
                        }
                        .padding(.top, 16)
                    }
                    .padding()
                }
            }
            .navigationTitle("User Profile")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

struct ApartmentChatView: View {
    let user: InterestedUser
    @State private var messageText = ""
    
    var body: some View {
        NavigationView {
            VStack {
                // Chat messages would go here
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        UserMessageBubble(message: user.initialMessage, isFromUser: false)
                    }
                    .padding()
                }
                
                // Message input
                HStack {
                    TextField("Type a message...", text: $messageText)
                        .padding(10)
                        .background(Color(.systemGray6))
                        .cornerRadius(20)
                    
                    Button(action: {}) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 30))
                            .foregroundColor(.blue)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
            }
            .navigationTitle(user.name)
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

struct UserMessageBubble: View {
    let message: String
    let isFromUser: Bool
    
    var body: some View {
        HStack {
            if isFromUser { Spacer() }
            
            Text(message)
                .padding(12)
                .background(isFromUser ? Color.blue : Color(.systemGray5))
                .foregroundColor(isFromUser ? .white : .primary)
                .cornerRadius(16)
                .frame(maxWidth: 280, alignment: isFromUser ? .trailing : .leading)
            
            if !isFromUser { Spacer() }
        }
    }
}

#Preview {
    InterestedUsersView()
} 