//
//  RootTabView.swift
//  RoommateSwipe
//
//  Created by AA on 2/27/25.
//

import SwiftUI

struct RootTabView: View {
    var body: some View {
        TabView {
            SwipeView()
                .tabItem {
                    Label("Home", systemImage: "house.fill")
                }

            LikedView()
                .tabItem {
                    Label("Liked", systemImage: "heart.fill")
                }

            MatchesView()
                .tabItem {
                    Label("Matches", systemImage: "person.2.fill")
                }

            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person.crop.circle")
                }
        }
    }
}

#Preview {
    RootTabView()
        .environmentObject(RoommateViewModel())
}
