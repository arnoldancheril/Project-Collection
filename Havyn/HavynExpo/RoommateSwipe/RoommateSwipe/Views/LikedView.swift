//
//  LikedView.swift
//  RoommateSwipe
//

import SwiftUI

struct LikedView: View {
    @EnvironmentObject var viewModel: RoommateViewModel

    var body: some View {
        NavigationView {
            ZStack {
                if viewModel.likedProfiles.isEmpty {
                    VStack(spacing: 20) {
                        Image(systemName: "heart.circle.fill")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 100, height: 100)
                            .foregroundColor(.pink.opacity(0.3))
                            .padding()
                            .background(
                                Circle()
                                    .fill(Color.pink.opacity(0.1))
                                    .frame(width: 160, height: 160)
                            )
                        
                        Text("No Likes Yet")
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        Text("Start swiping to find your perfect roommate match!")
                            .font(.body)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 40)
                        
                        Button {
                            // Switch to browse tab
                            if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                               let window = windowScene.windows.first,
                               let tabBarController = window.rootViewController as? UITabBarController {
                                tabBarController.selectedIndex = 0 // Browse tab
                            }
                        } label: {
                            HStack {
                                Image(systemName: "arrow.forward.circle.fill")
                                Text("Start Browsing")
                            }
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Color.blue)
                            .cornerRadius(25)
                        }
                        .padding(.top, 10)
                    }
                    .offset(y: -40) // Move up slightly for better visual balance
                } else {
                    List {
                        ForEach(viewModel.likedProfiles) { profile in
                            NavigationLink(destination: {
                                DetailedProfileView(
                                    profile: profile,
                                    onLike: {
                                        // Possibly do nothing since it's already liked
                                        // But if you want, you can re-insert it or just ignore
                                    },
                                    onDislike: {
                                        // Remove from liked list if user changes mind
                                        if let index = viewModel.likedProfiles.firstIndex(where: { $0.id == profile.id }) {
                                            viewModel.likedProfiles.remove(at: index)
                                        }
                                    }
                                )
                            }) {
                                HStack(spacing: 12) {
                                    Image(profile.imageName)
                                        .resizable()
                                        .scaledToFill()
                                        .frame(width: 60, height: 60)
                                        .clipShape(Circle())
                                    
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(profile.name)
                                            .font(.headline)
                                        Text("\(profile.age) • \(profile.city)")
                                            .font(.subheadline)
                                            .foregroundColor(.secondary)
                                    }
                                    
                                    Spacer()
                                    
                                    Image(systemName: "chevron.right")
                                        .foregroundColor(.secondary)
                                }
                                .padding(.vertical, 4)
                            }
                        }
                        .onDelete { indexSet in
                            viewModel.likedProfiles.remove(atOffsets: indexSet)
                        }
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("Liked Profiles")
        }
    }
}

#Preview {
    LikedView()
        .environmentObject(RoommateViewModel())
}
