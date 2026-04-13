//
//  ApartmentRootTabView.swift
//  RoommateSwipe
//

import SwiftUI

struct ApartmentRootTabView: View {
    @State private var isApartmentLister: Bool
    @State private var selectedTab = 0
    @Environment(\.dismiss) private var dismiss
    
    init(isApartmentLister: Bool) {
        self._isApartmentLister = State(initialValue: isApartmentLister)
    }
    
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Spacer()
                
                Text("Apartment Dashboard")
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Spacer()
            }
            .padding(.horizontal)
            .padding(.top, 8)
            .padding(.bottom, 4)
            
            TabView(selection: $selectedTab) {
                ListingsView()
                    .tabItem {
                        Label("Listings", systemImage: "building.2")
                    }
                    .tag(0)
                
                InterestedUsersView()
                    .tabItem {
                        Label("Interested", systemImage: "person.3")
                    }
                    .tag(1)
                
                AnalyticsView()
                    .tabItem {
                        Label("Analytics", systemImage: "chart.bar")
                    }
                    .tag(2)
                
                ApartmentSettingsView()
                    .tabItem {
                        Label("Settings", systemImage: "gear")
                    }
                    .tag(3)
            }
            .toolbarBackground(Color.white, for: .tabBar)
            .toolbarBackground(.visible, for: .tabBar)
        }
    }
}

#Preview {
    ApartmentRootTabView(isApartmentLister: true)
} 