//
//  SwipeView.swift
//  RoommateSwipe
//
//  Created by AA on 2/27/25.
//

import SwiftUI

struct SwipeView: View {
    @EnvironmentObject var viewModel: RoommateViewModel

    var body: some View {
        ZStack {
            // Background color (you could remove if you want the image behind safe areas)
            Color(.systemBackground)
                .ignoresSafeArea()

            GeometryReader { geo in
                ForEach(viewModel.profiles.indices, id: \.self) { index in
                    let profile = viewModel.profiles[index]
                    
                    SwipeCardView(profile: profile)
                        .frame(width: geo.size.width, height: geo.size.height)
                        .stacked(at: index, in: viewModel.profiles.count)
                        .gesture(
                            DragGesture()
                                .onEnded { value in
                                    if value.translation.width > 100 {
                                        // Swiped Right
                                        viewModel.like(profile: profile)
                                        viewModel.removeProfile(profile: profile)
                                    } else if value.translation.width < -100 {
                                        // Swiped Left
                                        viewModel.dislike(profile: profile)
                                        viewModel.removeProfile(profile: profile)
                                    }
                                }
                        )
                }
            }
        }
    }
}

// Keep the stacking offset if you want a layered look
extension View {
    func stacked(at index: Int, in total: Int) -> some View {
        let offset = Double(total - index) * 4
        return self.offset(x: 0, y: offset)
    }
}

#Preview {
    SwipeView()
        .environmentObject(RoommateViewModel())
}
