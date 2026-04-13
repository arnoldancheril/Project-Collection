//
//  SwipeCardView.swift
//  RoommateSwipe
//

import SwiftUI
import MapKit

struct SwipeCardView: View {
    let profile: Profile
    
    // Callbacks triggered by swiping
    let onSwipeLeft: () -> Void
    let onSwipeRight: () -> Void
    
    // For the property detail sheet
    @State private var showPropertySheet = false
    
    // For the profile detail sheet
    @State private var showDetail = false
    
    // Tracks the drag offset
    @State private var translation: CGSize = .zero
    private let swipeThreshold: CGFloat = 100
    
    // Animation states for feedback indicators
    @State private var showLikeIndicator = false
    @State private var showPassIndicator = false

    var body: some View {
        GeometryReader { geo in
            // ZStack that is fully swipable
            ZStack(alignment: .center) {
                
                // MAIN BACKGROUND: user's image + gradient
                Image(profile.imageName)
                    .resizable()
                    .scaledToFill()
                    .frame(width: geo.size.width, height: geo.size.height)
                    .clipped()
                
                LinearGradient(
                    gradient: Gradient(colors: [Color.clear, Color.black.opacity(0.6)]),
                    startPoint: .center,
                    endPoint: .bottom
                )
                .frame(width: geo.size.width, height: geo.size.height)
                
                // PROPERTY THUMBNAIL
                VStack {
                    HStack {
                        Spacer()
                        if let propertyImageName = profile.propertyImageName {
                            Image(propertyImageName)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 150, height: 100)
                                .clipped()
                                .cornerRadius(8)
                                .onTapGesture {
                                    showPropertySheet = true
                                }
                                .padding(.top, 10)
                                .padding(.trailing, 10)
                        }
                    }
                    Spacer()
                }
                
                // NAME, AGE, LOCATION, "MORE INFO" near bottom
                VStack(alignment: .leading, spacing: 6) {
                    Text("\(profile.name), \(profile.age)")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                    
                    Text(profile.city)
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    Button(action: {
                        showDetail = true
                    }) {
                        Text("More Info")
                            .fontWeight(.semibold)
                            .padding(.vertical, 6)
                            .padding(.horizontal, 12)
                            .background(Color.white.opacity(0.8))
                            .cornerRadius(8)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 80)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, geo.size.height * 0.65)
                
                // LIKE INDICATOR
                SwipeFeedbackLabel(
                    text: "LIKE",
                    color: .green,
                    rotationAngle: -30,
                    opacity: getLikeOpacity(for: translation.width, in: geo)
                )
                
                // PASS INDICATOR
                SwipeFeedbackLabel(
                    text: "PASS",
                    color: .red,
                    rotationAngle: 30,
                    opacity: getPassOpacity(for: translation.width, in: geo)
                )
                
                // Animated Like Indicator that appears on confirmed swipe
                if showLikeIndicator {
                    LikeAnimatedIndicator()
                        .transition(.scale.combined(with: .opacity))
                        .zIndex(10)
                }
                
                // Animated Pass Indicator that appears on confirmed swipe
                if showPassIndicator {
                    PassAnimatedIndicator()
                        .transition(.scale.combined(with: .opacity))
                        .zIndex(10)
                }
            }
            .contentShape(Rectangle()) // entire ZStack is swipable
            // Apply translation & rotation to the entire ZStack
            .offset(x: translation.width, y: translation.height)
            .rotationEffect(.degrees(Double(translation.width / geo.size.width) * 25), anchor: .bottom)
            .animation(.interactiveSpring(), value: translation)
            .gesture(
                DragGesture()
                    .onChanged { value in
                        withAnimation(.interactiveSpring()) {
                            translation = value.translation
                        }
                    }
                    .onEnded { value in
                        handleSwipeEnd(value: value, geo: geo)
                    }
            )
            // SHEET for the user's detailed profile
            .sheet(isPresented: $showDetail) {
                DetailedProfileView(
                    profile: profile,
                    onLike: { onSwipeRight() },
                    onDislike: { onSwipeLeft() }
                )
            }
            // SHEET for the property details
            .sheet(isPresented: $showPropertySheet) {
                DetailedPropertyView(profile: profile)
            }
        }
    }
    
    private func getLikeOpacity(for translation: CGFloat, in geo: GeometryProxy) -> Double {
        if translation <= 0 {
            return 0
        }
        
        // Scale from 0 to 1 based on how close we are to the threshold
        return min(1.0, Double(translation / swipeThreshold) * 0.8)
    }
    
    private func getPassOpacity(for translation: CGFloat, in geo: GeometryProxy) -> Double {
        if translation >= 0 {
            return 0
        }
        
        // Scale from 0 to 1 based on how close we are to the threshold
        return min(1.0, Double(-translation / swipeThreshold) * 0.8)
    }
    
    private func handleSwipeEnd(value: DragGesture.Value, geo: GeometryProxy) {
        if value.translation.width > swipeThreshold {
            // Swipe Right - LIKE
            withAnimation(.interactiveSpring()) {
                translation = CGSize(width: geo.size.width * 1.5, height: value.translation.height)
                showLikeIndicator = true
            }
            // Extend the delay to allow the animation to be more visible
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.7) {
                onSwipeRight()
            }
        } else if value.translation.width < -swipeThreshold {
            // Swipe Left - PASS
            withAnimation(.interactiveSpring()) {
                translation = CGSize(width: -geo.size.width * 1.5, height: value.translation.height)
                showPassIndicator = true
            }
            // Extend the delay to allow the animation to be more visible
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.7) {
                onSwipeLeft()
            }
        } else {
            // Snap back
            withAnimation(.interactiveSpring()) {
                translation = .zero
            }
        }
    }
}

// Swipe Feedback Label component
struct SwipeFeedbackLabel: View {
    let text: String
    let color: Color
    let rotationAngle: Double
    let opacity: Double
    
    var body: some View {
        Text(text)
            .font(.system(size: 42, weight: .black))
            .foregroundColor(color)
            .padding(20)
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(color, lineWidth: 4)
            )
            .background(Color.white.opacity(0.2))
            .cornerRadius(10)
            .rotationEffect(.degrees(rotationAngle))
            .opacity(opacity)
            .padding(40)
    }
}

// Like Animation - Appears when user confirms a right swipe
struct LikeAnimatedIndicator: View {
    @State private var scale: CGFloat = 0.5
    @State private var opacity: Double = 0
    
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "heart.fill")
                .font(.system(size: 100))
                .foregroundColor(.green)
            
            Text("LIKE")
                .font(.system(size: 48, weight: .heavy))
                .foregroundColor(.white)
        }
        .padding(40)
        .background(Color.green.opacity(0.4))
        .cornerRadius(25)
        .overlay(
            RoundedRectangle(cornerRadius: 25)
                .stroke(Color.green, lineWidth: 5)
        )
        .rotationEffect(.degrees(-15))
        .shadow(color: Color.green.opacity(0.5), radius: 15, x: 0, y: 8)
        .scaleEffect(scale)
        .opacity(opacity)
        .onAppear {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                scale = 1.1
                opacity = 1
            }
            withAnimation(.spring(response: 0.4, dampingFraction: 0.7).delay(0.1)) {
                scale = 1.0
            }
        }
    }
}

// Pass Animation - Appears when user confirms a left swipe
struct PassAnimatedIndicator: View {
    @State private var scale: CGFloat = 0.5
    @State private var opacity: Double = 0
    
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "xmark.circle.fill")
                .font(.system(size: 100))
                .foregroundColor(.red)
            
            Text("PASS")
                .font(.system(size: 48, weight: .heavy))
                .foregroundColor(.white)
        }
        .padding(40)
        .background(Color.red.opacity(0.4))
        .cornerRadius(25)
        .overlay(
            RoundedRectangle(cornerRadius: 25)
                .stroke(Color.red, lineWidth: 5)
        )
        .rotationEffect(.degrees(15))
        .shadow(color: Color.red.opacity(0.5), radius: 15, x: 0, y: 8)
        .scaleEffect(scale)
        .opacity(opacity)
        .onAppear {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                scale = 1.1
                opacity = 1
            }
            withAnimation(.spring(response: 0.4, dampingFraction: 0.7).delay(0.1)) {
                scale = 1.0
            }
        }
    }
}

#Preview {
    SwipeCardView(
        profile: Profile(
            name: "Aladdin",
            age: 22,
            gender: "Male",
            city: "Chicago",
            bio: "Dog person, weekend hiker.",
            imageName: "exampleProfile4",
            propertyImageName: "exampleProperty4",
            hasRoom: true,
            needsRoom: false,
            moveInDate: Date().addingTimeInterval(30 * 24 * 60 * 60), // 30 days from now
            preferredNeighborhoods: ["Wicker Park", "Logan Square"],
            budgetRange: 1000...1500,
            coordinate: CLLocationCoordinate2D(latitude: 41.9088, longitude: -87.6796), // Wicker Park
            numberOfRooms: 2,
            numberOfBathrooms: 2,
            amenities: "Covered parking, Storage",
            rent: "$1100 / month",
            address: "1550 N Milwaukee Ave, Chicago, IL 60622",
            cleanliness: 4,
            partying: 2,
            smoking: false,
            pets: true,
            petTypes: ["Abu"],
            wakeUpTime: "6:00 AM",
            sleepTime: "10:00 PM",
            habits: "Clean, early riser, occasionally blasts Disney tunes.",
            lookingFor: "Friendly, active roommate who loves the outdoors.",
            verificationStatus: true,
            isBlocked: false
        ),
        onSwipeLeft: { },
        onSwipeRight: { }
    )
}
