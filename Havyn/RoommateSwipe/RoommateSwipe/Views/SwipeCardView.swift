//
//  SwipeCardView.swift
//  RoommateSwipe
//
//  Created by AA on 2/27/25.
//

import SwiftUI

struct SwipeCardView: View {
    let profile: Profile

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .bottomLeading) {
                // The full-screen background image
                Image(profile.imageName)
                    .resizable()
                    .scaledToFill()
                    .frame(width: geo.size.width, height: geo.size.height)
                    .clipped()

                // A gradient overlay to make text more visible
                LinearGradient(
                    gradient: Gradient(colors: [Color.clear, Color.black.opacity(0.6)]),
                    startPoint: .center,
                    endPoint: .bottom
                )
                .frame(width: geo.size.width, height: geo.size.height)

                // Name, city, bio
                VStack(alignment: .leading, spacing: 6) {
                    Text("\(profile.name), \(profile.age)")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(.white)

                    Text(profile.city)
                        .font(.title2)
                        .foregroundColor(.white)

                    Text(profile.bio)
                        .font(.subheadline)
                        .foregroundColor(.white)
                        .lineLimit(3)
                }
                .padding(.horizontal, 16)
                // Increase the bottom padding to sit just above the tab bar
                .padding(.bottom, 60)
            }
        }
    }
}

#Preview {
    SwipeCardView(profile: Profile(
        name: "Taylor",
        age: 28,
        city: "Seattle",
        bio: "Dog person, weekend hiker.",
        imageName: "exampleProfile4"
    ))
}
