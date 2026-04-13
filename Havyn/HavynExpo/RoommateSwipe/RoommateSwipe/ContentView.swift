//
//  ContentView.swift
//  RoommateSwipe
//
//  Created by AA on 2/27/25.
//


import SwiftUI
import SwiftData

struct ContentView: View {
    // MARK: - SwiftData environment
    @Environment(\.modelContext) private var modelContext
    @Query private var items: [Item]
    
    // MARK: - Loading State
    @State private var isLoading: Bool = true
    @State private var isAuthenticated: Bool = false
    @State private var isApartmentLister: Bool = false
    
    // MARK: - ViewModel
    @StateObject private var viewModel = RoommateViewModel()
    
    var body: some View {
        ZStack {
            if isLoading {
                LoadingView()
                    .transition(.opacity.combined(with: .scale))
            } else if !isAuthenticated && !isApartmentLister {
                LoginView(isAuthenticated: $isAuthenticated, isApartmentLister: $isApartmentLister)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            } else if isApartmentLister {
                // The apartment lister UI
                ApartmentRootTabView(isApartmentLister: true)
                    .transition(.asymmetric(
                        insertion: .move(edge: .trailing).combined(with: .opacity),
                        removal: .move(edge: .leading).combined(with: .opacity)
                    ))
            } else {
                // The main UI with tabs for regular users
                RootTabView(isAuthenticated: $isAuthenticated)
                    .environmentObject(viewModel)
                    .transition(.asymmetric(
                        insertion: .move(edge: .trailing).combined(with: .opacity),
                        removal: .move(edge: .leading).combined(with: .opacity)
                    ))
            }
        }
        .animation(.spring(duration: 0.5), value: isLoading)
        .animation(.spring(duration: 0.5), value: isAuthenticated)
        .animation(.spring(duration: 0.5), value: isApartmentLister)
        .onAppear {
            // Simulate a 2-second loading time
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                withAnimation {
                    isLoading = false
                }
            }
        }
    }
    
    // You can keep or remove the default SwiftData logic below:
    /*
    private func addItem() {
        withAnimation {
            let newItem = Item(timestamp: Date())
            modelContext.insert(newItem)
        }
    }

    private func deleteItems(offsets: IndexSet) {
        withAnimation {
            for index in offsets {
                modelContext.delete(items[index])
            }
        }
    }
    */
}

#Preview {
    ContentView()
        .modelContainer(for: Item.self, inMemory: true)
}
