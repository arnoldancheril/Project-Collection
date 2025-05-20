//
//  LoadingView.swift
//  RoommateSwipe
//
//  Created by AA on 2/27/25.
//

import SwiftUI

struct LoadingView: View {
    var body: some View {
        ZStack {
            Color.white.ignoresSafeArea()
            VStack(spacing: 20) {
                // Placeholder logo (replace with your custom logo in Assets if you have one)
                Image(systemName: "house.circle.fill")
                    .resizable()
                    .frame(width: 80, height: 80)
                    .foregroundColor(.pink)
                
                Text("Crib")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .foregroundColor(.pink)
                
                ProgressView("Finding your perfect roommate...")
                    .padding()
            }
        }
    }
}

#Preview {
    LoadingView()
}
