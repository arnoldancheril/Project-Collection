//
//  LikedView.swift
//  RoommateSwipe
//
//  Created by AA on 2/27/25.
//

import SwiftUI

struct LikedView: View {
    @EnvironmentObject var viewModel: RoommateViewModel

    var body: some View {
        NavigationView {
            List(viewModel.likedProfiles) { profile in
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
            .navigationTitle("Liked Profiles")
        }
    }
}

#Preview {
    LikedView()
        .environmentObject(RoommateViewModel())
}
