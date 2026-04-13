//
//  MatchesView.swift
//  RoommateSwipe
//

import SwiftUI

struct MatchesView: View {
    @EnvironmentObject var viewModel: RoommateViewModel
    @State private var selectedTab = 0
    @State private var showCreateGroup = false
    
    // Demo data - in a real app, this would come from the backend
    @State private var groupChats = [
        GroupChat(id: "1", name: "Chicago Downtown Apartments", members: 8, lastMessage: "Has anyone toured the units on Michigan Ave?", lastActivity: "10m ago"),
        GroupChat(id: "2", name: "Budget Friendly Options", members: 5, lastMessage: "I found a great deal in Wicker Park", lastActivity: "2h ago"),
        GroupChat(id: "3", name: "Lincoln Park Roommates", members: 6, lastMessage: "Looking for 2 more people to join our apartment", lastActivity: "Yesterday")
    ]

    var body: some View {
        NavigationView {
            VStack {
                // Tab segmentation
                Picker("View", selection: $selectedTab) {
                    Text("Matches").tag(0)
                    Text("Group Chats").tag(1)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)
                
                if selectedTab == 0 {
                    // Matches List
                    List {
                        ForEach(viewModel.matchedProfiles) { profile in
                            NavigationLink(destination: {
                                MatchedProfileView(profile: profile)
                            }) {
                                HStack {
                                    Image(profile.imageName)
                                        .resizable()
                                        .scaledToFill()
                                        .frame(width: 50, height: 50)
                                        .clipShape(Circle())

                                    VStack(alignment: .leading) {
                                        Text(profile.name)
                                            .font(.headline)
                                        Text("\(profile.age), \(profile.city)")
                                            .font(.subheadline)
                                            .foregroundColor(.secondary)
                                    }
                                }
                            }
                        }
                    }
                    .overlay {
                        if viewModel.matchedProfiles.isEmpty {
                            ContentUnavailableView(
                                "No Matches Yet",
                                systemImage: "person.2.slash",
                                description: Text("Keep swiping to find your perfect roommate!")
                            )
                        }
                    }
                } else {
                    // Group Chats View
                    ScrollView {
                        VStack(spacing: 16) {
                            // Create New Group Button
                            Button(action: {
                                showCreateGroup = true
                            }) {
                                HStack {
                                    Image(systemName: "plus.circle.fill")
                                        .font(.title2)
                                        .foregroundColor(.white)
                                    Text("Create New Group Chat")
                                        .fontWeight(.semibold)
                                        .foregroundColor(.white)
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(
                                    LinearGradient(
                                        gradient: Gradient(colors: [Color.blue, Color.purple]),
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .cornerRadius(12)
                                .shadow(color: Color.blue.opacity(0.3), radius: 5)
                            }
                            .padding(.horizontal)
                            .padding(.top)
                            
                            // Explore Group Chats
                            GroupChatSection(title: "Your Group Chats", groups: groupChats.filter { $0.id == "1" })
                                .padding(.horizontal)
                            
                            GroupChatSection(title: "Suggested Group Chats", groups: groupChats.filter { $0.id != "1" })
                                .padding(.horizontal)
                            
                            // Infographic
                            VStack(alignment: .leading, spacing: 12) {
                                Text("Why Join Group Chats?")
                                    .font(.headline)
                                    .foregroundColor(.primary)
                                
                                GroupChatFeatureRow(
                                    icon: "building.2.fill",
                                    title: "Discuss Properties",
                                    description: "Share insights about different apartments and neighborhoods"
                                )
                                
                                GroupChatFeatureRow(
                                    icon: "person.3.fill",
                                    title: "Find Roommates",
                                    description: "Connect with potential roommates for specific properties"
                                )
                                
                                GroupChatFeatureRow(
                                    icon: "calendar",
                                    title: "Coordinate Viewings",
                                    description: "Plan apartment tours together with other interested users"
                                )
                            }
                            .padding()
                            .background(Color(.secondarySystemBackground))
                            .cornerRadius(12)
                            .padding(.horizontal)
                        }
                        .padding(.bottom)
                    }
                    .overlay {
                        if groupChats.isEmpty {
                            ContentUnavailableView(
                                "No Group Chats Yet",
                                systemImage: "bubble.left.and.bubble.right",
                                description: Text("Create a group chat to discuss properties with others")
                            )
                        }
                    }
                }
            }
            .navigationTitle(selectedTab == 0 ? "Matches" : "Group Chats")
            .sheet(isPresented: $showCreateGroup) {
                CreateGroupView(onCreated: { newGroup in
                    groupChats.append(newGroup)
                    showCreateGroup = false
                })
            }
        }
    }
}

// Group Chat Model for Demo
struct GroupChat: Identifiable {
    var id: String
    var name: String
    var members: Int
    var lastMessage: String
    var lastActivity: String
}

// Group Chat Section
struct GroupChatSection: View {
    let title: String
    let groups: [GroupChat]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
                .padding(.leading, 4)
            
            if groups.isEmpty {
                Text("No group chats available")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .padding()
            } else {
                ForEach(groups) { group in
                    NavigationLink(destination: GroupChatDetailView(group: group)) {
                        GroupChatRow(group: group)
                    }
                    .buttonStyle(PlainButtonStyle())
                }
            }
        }
    }
}

// Group Chat Row Component
struct GroupChatRow: View {
    let group: GroupChat
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(group.name)
                    .font(.headline)
                
                Spacer()
                
                Text("\(group.members) members")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(group.lastMessage)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .lineLimit(1)
            
            HStack {
                Spacer()
                
                Text(group.lastActivity)
                    .font(.caption)
                    .foregroundColor(.gray)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }
}

// Feature Row for Infographic
struct GroupChatFeatureRow: View {
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 24))
                .foregroundColor(.blue)
                .frame(width: 32, height: 32)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

// Group Chat Detail View
struct GroupChatDetailView: View {
    let group: GroupChat
    @State private var message = ""
    
    // Demo messages - in a real app, these would come from a database
    let messages = [
        ChatMessage(id: "1", sender: "Michael", text: "Has anyone toured the units on Michigan Ave?", time: "10:30 AM", isCurrentUser: false),
        ChatMessage(id: "2", sender: "Sarah", text: "Yes, I went yesterday. The 2BR units are really nice but a bit pricey.", time: "10:35 AM", isCurrentUser: false),
        ChatMessage(id: "3", sender: "You", text: "What was the price range?", time: "10:37 AM", isCurrentUser: true),
        ChatMessage(id: "4", sender: "Sarah", text: "Around $2,500-3,000 for the 2BR units.", time: "10:40 AM", isCurrentUser: false),
        ChatMessage(id: "5", sender: "Michael", text: "That's actually not bad for that location! I was expecting worse.", time: "10:42 AM", isCurrentUser: false),
        ChatMessage(id: "6", sender: "You", text: "I'm interested in splitting a unit. Anyone looking for a roommate?", time: "10:45 AM", isCurrentUser: true)
    ]
    
    var body: some View {
        VStack {
            // Chat header with info
            VStack {
                Text(group.name)
                    .font(.headline)
                
                Text("\(group.members) members")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color(.secondarySystemBackground))
            
            // Messages
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(messages) { message in
                        ChatBubble(message: message)
                            .padding(EdgeInsets(top: 0, leading: 16, bottom: 0, trailing: 16))
                    }
                }
                .padding(.vertical)
            }
            
            // Message input
            HStack {
                TextField("Type a message...", text: $message)
                    .padding(10)
                    .background(Color(.tertiarySystemBackground))
                    .cornerRadius(20)
                
                Button(action: {
                    // Send message action would go here
                    message = ""
                }) {
                    Image(systemName: "paperplane.fill")
                        .foregroundColor(.blue)
                        .padding(10)
                }
            }
            .padding()
            .background(Color(.secondarySystemBackground))
        }
        .navigationTitle("Group Chat")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: {
                    // Info action would go here
                }) {
                    Image(systemName: "info.circle")
                }
            }
        }
    }
}

// Create Group View
struct CreateGroupView: View {
    @Environment(\.dismiss) var dismiss
    @State private var groupName = ""
    @State private var groupDescription = ""
    @State private var isPublic = true
    var onCreated: (GroupChat) -> Void
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Group Details")) {
                    TextField("Group Name", text: $groupName)
                    
                    TextField("Group Description", text: $groupDescription)
                        .frame(height: 80)
                }
                
                Section(header: Text("Privacy")) {
                    Toggle("Public Group", isOn: $isPublic)
                    
                    if !isPublic {
                        Text("Only invited members can join")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    } else {
                        Text("Anyone can find and join this group")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Section(header: Text("Focus")) {
                    Text("This group will focus on discussing:")
                        .font(.subheadline)
                    
                    ForEach(["Properties", "Roommates", "Neighborhoods", "Pricing", "Amenities"], id: \.self) { focus in
                        HStack {
                            Text(focus)
                            Spacer()
                            Image(systemName: "checkmark")
                                .foregroundColor(.blue)
                                .opacity(focus == "Properties" || focus == "Roommates" ? 1 : 0)
                        }
                    }
                }
            }
            .navigationTitle("Create Group Chat")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                leading: Button("Cancel") {
                    dismiss()
                },
                trailing: Button("Create") {
                    let newGroup = GroupChat(
                        id: UUID().uuidString,
                        name: groupName,
                        members: 1,
                        lastMessage: "Group created",
                        lastActivity: "Just now"
                    )
                    onCreated(newGroup)
                }
                .disabled(groupName.isEmpty)
            )
        }
    }
}

#Preview {
    MatchesView()
        .environmentObject(RoommateViewModel())
}
