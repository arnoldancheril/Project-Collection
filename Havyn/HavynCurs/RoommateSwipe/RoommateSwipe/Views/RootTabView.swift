//
//  RootTabView.swift
//  RoommateSwipe
//

import SwiftUI

struct RootTabView: View {
    @Binding var isAuthenticated: Bool
    @EnvironmentObject var viewModel: RoommateViewModel
    
    var body: some View {
        TabView {
            SwipeView()
                .tabItem {
                    Label("Browse", systemImage: "square.stack")
                }
            
            MapView()
                .tabItem {
                    Label("Map", systemImage: "map")
                }
            
            LikedView()
                .tabItem {
                    Label("Liked", systemImage: "heart")
                }
            
            MatchesView()
                .tabItem {
                    Label("Matches", systemImage: "person.2")
                }
            
            ProfileDisplayView(isAuthenticated: $isAuthenticated)
                .tabItem {
                    Label("Profile", systemImage: "person.circle")
                }
        }
        .toolbarBackground(Color.white, for: .tabBar)
        .toolbarBackground(.visible, for: .tabBar)
    }
}

#Preview {
    RootTabView(isAuthenticated: .constant(true))
        .environmentObject(RoommateViewModel())
}
