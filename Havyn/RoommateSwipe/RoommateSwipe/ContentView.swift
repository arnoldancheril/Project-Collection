//
//  ContentView.swift
//  RoommateSwipe
//
//  Created by AA on 2/27/25.
//

import SwiftUI
import SwiftData

struct ContentView: View {
    // MARK: - SwiftData environment (if you still want to keep it)
    @Environment(\.modelContext) private var modelContext
    @Query private var items: [Item]
    
    // MARK: - Loading State
    @State private var isLoading: Bool = true
    
    // MARK: - ViewModel
    @StateObject var viewModel = RoommateViewModel()
    
    var body: some View {
        Group {
            if isLoading {
                LoadingView()
                    .transition(.opacity)
            } else {
                // The main UI with tabs
                RootTabView()
                    .environmentObject(viewModel)
                    .transition(.opacity)
            }
        }
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
