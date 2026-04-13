//
//  MatchedProfileView.swift
//  RoommateSwipe
//

import SwiftUI
import MapKit

struct MatchedProfileView: View {
    let profile: Profile
    @EnvironmentObject var viewModel: RoommateViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showPropertySheet = false
    @State private var showUnmatchAlert = false
    @State private var showChat = false
    @State private var messageText = ""
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Profile Image
                    Image(profile.imageName)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity)
                        .clipped()

                    VStack(alignment: .leading, spacing: 16) {
                        // Basic Info
                        VStack(alignment: .leading, spacing: 8) {
                            Text("\(profile.name), \(profile.age)")
                                .font(.title)
                                .fontWeight(.bold)

                            Text(profile.city)
                                .font(.title3)
                                .foregroundColor(.secondary)

                            Text(profile.bio)
                                .font(.body)
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(.top, 4)
                        }
                        .padding(.horizontal)

                        Divider()
                            .padding(.horizontal)

                        // Habits Section
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Habits")
                                .font(.title2)
                                .fontWeight(.semibold)
                            Text(profile.habits)
                                .font(.body)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(.horizontal)

                        Divider()
                            .padding(.horizontal)

                        // Looking For Section
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Looking For")
                                .font(.title2)
                                .fontWeight(.semibold)
                            Text(profile.lookingFor)
                                .font(.body)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(.horizontal)

                        Divider()
                            .padding(.horizontal)

                        // Home Details Section
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Home Details")
                                .font(.title2)
                                .fontWeight(.semibold)

                            VStack(alignment: .leading, spacing: 8) {
                                Text("Rooms: \(profile.numberOfRooms)")
                                Text("Bathrooms: \(profile.numberOfBathrooms)")
                                Text("Rent: \(profile.rent)")
                                    .foregroundColor(.blue)
                                Text("Amenities: \(profile.amenities)")
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .font(.body)
                        }
                        .padding(.horizontal)

                        // Property Info Button
                        Button {
                            showPropertySheet = true
                        } label: {
                            Text("View Property Details")
                                .fontWeight(.semibold)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue.opacity(0.2))
                                .cornerRadius(10)
                        }
                        .padding(.horizontal)
                        .padding(.top, 8)
                    }
                }
            }
            .navigationTitle("Match Profile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button(role: .destructive) {
                            showUnmatchAlert = true
                        } label: {
                            Label("Unmatch", systemImage: "person.badge.minus")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .safeAreaInset(edge: .bottom) {
                VStack {
                    // Chat Button
                    Button {
                        showChat = true
                    } label: {
                        Text("Message")
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .foregroundColor(.white)
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    .padding()
                    .background(.thinMaterial)
                }
            }
            .sheet(isPresented: $showPropertySheet) {
                DetailedPropertyView(profile: profile)
            }
            .sheet(isPresented: $showChat) {
                ChatView(profile: profile)
            }
            .alert("Unmatch from \(profile.name)?", isPresented: $showUnmatchAlert) {
                Button("Cancel", role: .cancel) { }
                Button("Unmatch", role: .destructive) {
                    if let index = viewModel.matchedProfiles.firstIndex(where: { $0.id == profile.id }) {
                        viewModel.matchedProfiles.remove(at: index)
                        dismiss()
                    }
                }
            } message: {
                Text("This will remove the match and delete your chat history.")
            }
        }
    }
}

struct ChatView: View {
    let profile: Profile
    @Environment(\.dismiss) private var dismiss
    @State private var messageText = ""
    @State private var messages: [ChatMessage] = []
    
    var body: some View {
        NavigationView {
            VStack {
                // Chat messages
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(messages) { message in
                            ChatBubble(message: message)
                                .padding(EdgeInsets(top: 0, leading: 16, bottom: 0, trailing: 16))
                        }
                    }
                    .padding(.top)
                }
                
                // Message input
                HStack {
                    TextField("Type a message...", text: $messageText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .padding(.horizontal)
                    
                    Button {
                        sendMessage()
                    } label: {
                        Image(systemName: "paperplane.fill")
                            .foregroundColor(.blue)
                    }
                    .padding(.trailing)
                    .disabled(messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .padding(.vertical, 8)
                .background(.thinMaterial)
            }
            .navigationTitle("Chat with \(profile.name)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func sendMessage() {
        let trimmedText = messageText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return }
        
        let newMessage = ChatMessage(
            id: UUID().uuidString,
            sender: "You",
            text: trimmedText,
            time: formatTime(Date()),
            isCurrentUser: true
        )
        messages.append(newMessage)
        messageText = ""
        
        // Simulate a response after a short delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            let response = ChatMessage(
                id: UUID().uuidString,
                sender: profile.name,
                text: "Thanks for your message! I'll get back to you soon.",
                time: formatTime(Date()),
                isCurrentUser: false
            )
            messages.append(response)
        }
    }
    
    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

#Preview {
    MatchedProfileView(
        profile: Profile(
            name: "Aladdin",
            age: 22,
            gender: "Male",
            city: "Chicago",
            bio: "Dog person, weekend hiker.",
            imageName: "exampleProfile4",
            propertyImageName: "exampleProperty4",
            hasRoom: true,
            needsRoom: false,
            moveInDate: Date().addingTimeInterval(30 * 24 * 60 * 60), // 30 days from now
            preferredNeighborhoods: ["Wicker Park", "Logan Square"],
            budgetRange: 1000...1500,
            coordinate: CLLocationCoordinate2D(latitude: 41.9088, longitude: -87.6796), // Wicker Park
            numberOfRooms: 2,
            numberOfBathrooms: 2,
            amenities: "Covered parking, Storage",
            rent: "$1100 / month",
            address: "1550 N Milwaukee Ave, Chicago, IL 60622",
            cleanliness: 4,
            partying: 2,
            smoking: false,
            pets: true,
            petTypes: ["Abu"],
            wakeUpTime: "6:00 AM",
            sleepTime: "10:00 PM",
            habits: "Clean, early riser, occasionally blasts Disney tunes.",
            lookingFor: "Friendly, active roommate who loves the outdoors.",
            verificationStatus: true,
            isBlocked: false
        )
    )
    .environmentObject(RoommateViewModel())
} 