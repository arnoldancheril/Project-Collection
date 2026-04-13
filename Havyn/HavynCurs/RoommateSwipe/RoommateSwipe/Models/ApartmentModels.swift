//
//  ApartmentModels.swift
//  RoommateSwipe
//

import Foundation
import SwiftUI
import CoreLocation
import Firebase
import FirebaseFirestore

// MARK: - Apartment Listing
struct ApartmentListing: Identifiable, Hashable {
    var id: String
    var ownerId: String
    var name: String
    var address: String
    var city: String
    var state: String
    var zipCode: String
    var description: String
    var monthlyRent: Double
    var bedrooms: Int
    var bathrooms: Double
    var squareFootage: Int
    var availableDate: Date
    var leaseLength: [LeaseLength]
    var amenities: [String]
    var petPolicy: PetPolicy
    var images: [String] // URLs to images
    var coordinates: CLLocationCoordinate2D
    var isActive: Bool
    var dateCreated: Date
    var dateModified: Date
    var interestedUsers: [String] // IDs of interested users
    
    // Custom hash function for CLLocationCoordinate2D
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(name)
        hasher.combine(monthlyRent)
        hasher.combine(coordinates.latitude)
        hasher.combine(coordinates.longitude)
    }
    
    static func == (lhs: ApartmentListing, rhs: ApartmentListing) -> Bool {
        return lhs.id == rhs.id
    }
    
    // Convert Firebase document to ApartmentListing
    static func fromFirestore(document: DocumentSnapshot) -> ApartmentListing? {
        guard let data = document.data() else { return nil }
        
        guard let ownerId = data["ownerId"] as? String,
              let name = data["name"] as? String,
              let address = data["address"] as? String,
              let city = data["city"] as? String,
              let state = data["state"] as? String,
              let zipCode = data["zipCode"] as? String,
              let description = data["description"] as? String,
              let monthlyRent = data["monthlyRent"] as? Double,
              let bedrooms = data["bedrooms"] as? Int,
              let bathrooms = data["bathrooms"] as? Double,
              let squareFootage = data["squareFootage"] as? Int,
              let availableDateTimestamp = data["availableDate"] as? Timestamp,
              let leaseLengthStrings = data["leaseLength"] as? [String],
              let amenities = data["amenities"] as? [String],
              let petPolicyString = data["petPolicy"] as? String,
              let images = data["images"] as? [String],
              let geoPoint = data["coordinates"] as? GeoPoint,
              let isActive = data["isActive"] as? Bool,
              let createdTimestamp = data["dateCreated"] as? Timestamp,
              let modifiedTimestamp = data["dateModified"] as? Timestamp,
              let interestedUsers = data["interestedUsers"] as? [String]
        else {
            return nil
        }
        
        let coordinates = CLLocationCoordinate2D(
            latitude: geoPoint.latitude,
            longitude: geoPoint.longitude
        )
        
        let petPolicy = PetPolicy(rawValue: petPolicyString) ?? .noPets
        let leaseLength = leaseLengthStrings.compactMap { LeaseLength(rawValue: $0) }
        
        return ApartmentListing(
            id: document.documentID,
            ownerId: ownerId,
            name: name,
            address: address,
            city: city,
            state: state,
            zipCode: zipCode,
            description: description,
            monthlyRent: monthlyRent,
            bedrooms: bedrooms,
            bathrooms: bathrooms,
            squareFootage: squareFootage,
            availableDate: availableDateTimestamp.dateValue(),
            leaseLength: leaseLength,
            amenities: amenities,
            petPolicy: petPolicy,
            images: images,
            coordinates: coordinates,
            isActive: isActive,
            dateCreated: createdTimestamp.dateValue(),
            dateModified: modifiedTimestamp.dateValue(),
            interestedUsers: interestedUsers
        )
    }
    
    // Convert ApartmentListing to Firebase document data
    func toFirestore() -> [String: Any] {
        let geoPoint = GeoPoint(latitude: coordinates.latitude, longitude: coordinates.longitude)
        let leaseLengthStrings = leaseLength.map { $0.rawValue }
        
        return [
            "ownerId": ownerId,
            "name": name,
            "address": address,
            "city": city,
            "state": state,
            "zipCode": zipCode,
            "description": description,
            "monthlyRent": monthlyRent,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "squareFootage": squareFootage,
            "availableDate": Timestamp(date: availableDate),
            "leaseLength": leaseLengthStrings,
            "amenities": amenities,
            "petPolicy": petPolicy.rawValue,
            "images": images,
            "coordinates": geoPoint,
            "isActive": isActive,
            "dateCreated": Timestamp(date: dateCreated),
            "dateModified": Timestamp(date: Date()),
            "interestedUsers": interestedUsers
        ]
    }
    
    static func sampleListings() -> [ApartmentListing] {
        return [
            ApartmentListing(
                id: "1",
                ownerId: "owner1",
                name: "Riverside Apartments",
                address: "123 Main St",
                city: "New York",
                state: "NY",
                zipCode: "10001",
                description: "Beautiful apartment with river views, modern finishes, and spacious layout. Close to parks, shopping, and public transportation.",
                monthlyRent: 2200,
                bedrooms: 2,
                bathrooms: 2,
                squareFootage: 1050,
                availableDate: Date().addingTimeInterval(86400 * 14), // 2 weeks from now
                leaseLength: [.sixMonths, .oneYear],
                amenities: ["In-unit Laundry", "Dishwasher", "Central AC", "Fitness Center", "Rooftop Deck"],
                petPolicy: .petsAllowed,
                images: ["apartment1_1", "apartment1_2", "apartment1_3"],
                coordinates: CLLocationCoordinate2D(latitude: 40.7128, longitude: -74.0060),
                isActive: true,
                dateCreated: Date().addingTimeInterval(-86400 * 30), // 30 days ago
                dateModified: Date(),
                interestedUsers: ["user1", "user2", "user3"]
            ),
            ApartmentListing(
                id: "2",
                ownerId: "owner1",
                name: "Downtown Lofts",
                address: "456 Broadway",
                city: "New York",
                state: "NY",
                zipCode: "10012",
                description: "Stunning loft in the heart of downtown. High ceilings, exposed brick, and large windows. Walking distance to restaurants and attractions.",
                monthlyRent: 3100,
                bedrooms: 1,
                bathrooms: 1.5,
                squareFootage: 950,
                availableDate: Date().addingTimeInterval(86400 * 7), // 1 week from now
                leaseLength: [.oneYear, .twoYears],
                amenities: ["Stainless Steel Appliances", "Hardwood Floors", "24/7 Doorman", "Package Room", "Bike Storage"],
                petPolicy: .smallPetsOnly,
                images: ["apartment2_1", "apartment2_2"],
                coordinates: CLLocationCoordinate2D(latitude: 40.7215, longitude: -73.9968),
                isActive: true,
                dateCreated: Date().addingTimeInterval(-86400 * 15), // 15 days ago
                dateModified: Date(),
                interestedUsers: ["user4"]
            )
        ]
    }
}

// MARK: - Pet Policy
enum PetPolicy: String, CaseIterable {
    case noPets = "No Pets"
    case smallPetsOnly = "Small Pets Only"
    case petsAllowed = "Pets Allowed"
    case petsWithDeposit = "Pets with Deposit"
    
    var icon: String {
        switch self {
        case .noPets: return "pawprint.slash"
        case .smallPetsOnly: return "pawprint"
        case .petsAllowed: return "pawprint.circle.fill"
        case .petsWithDeposit: return "pawprint.circle"
        }
    }
}

// MARK: - Lease Length
enum LeaseLength: String, CaseIterable {
    case threeMonths = "3 Months"
    case sixMonths = "6 Months"
    case nineMonths = "9 Months"
    case oneYear = "1 Year"
    case twoYears = "2 Years"
    case flexible = "Flexible"
}

// MARK: - Interested User
struct InterestedUser: Identifiable, Hashable {
    var id: String
    var name: String
    var age: Int
    var occupation: String
    var budget: Double
    var moveInDate: Date
    var profileImage: String? // URL to image
    var initialMessage: String
    var listingId: String
    var dateInterested: Date
    var messages: [Message]
    var status: InterestStatus
    
    static func == (lhs: InterestedUser, rhs: InterestedUser) -> Bool {
        return lhs.id == rhs.id && lhs.listingId == rhs.listingId
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(listingId)
    }
    
    // Convert Firebase document to InterestedUser
    static func fromFirestore(document: DocumentSnapshot) -> InterestedUser? {
        guard let data = document.data() else { return nil }
        
        guard let name = data["name"] as? String,
              let age = data["age"] as? Int,
              let occupation = data["occupation"] as? String,
              let budget = data["budget"] as? Double,
              let moveInTimestamp = data["moveInDate"] as? Timestamp,
              let listingId = data["listingId"] as? String,
              let interestedTimestamp = data["dateInterested"] as? Timestamp,
              let statusString = data["status"] as? String
        else {
            return nil
        }
        
        let profileImage = data["profileImage"] as? String
        let initialMessage = data["initialMessage"] as? String ?? "I'm interested in this property!"
        let status = InterestStatus(rawValue: statusString) ?? .new
        
        // Convert message array if it exists
        var messages: [Message] = []
        if let messagesData = data["messages"] as? [[String: Any]] {
            messages = messagesData.compactMap { messageDict -> Message? in
                guard let text = messageDict["text"] as? String,
                      let senderIdString = messageDict["senderId"] as? String,
                      let timestampData = messageDict["timestamp"] as? Timestamp,
                      let isFromLister = messageDict["isFromLister"] as? Bool
                else {
                    return nil
                }
                
                let senderId = senderIdString
                
                return Message(
                    id: UUID().uuidString,
                    text: text,
                    senderId: senderId,
                    timestamp: timestampData.dateValue(),
                    isFromLister: isFromLister
                )
            }
        }
        
        return InterestedUser(
            id: document.documentID,
            name: name,
            age: age,
            occupation: occupation,
            budget: budget,
            moveInDate: moveInTimestamp.dateValue(),
            profileImage: profileImage,
            initialMessage: initialMessage,
            listingId: listingId,
            dateInterested: interestedTimestamp.dateValue(),
            messages: messages,
            status: status
        )
    }
    
    // Convert InterestedUser to Firebase document data
    func toFirestore() -> [String: Any] {
        let messagesData = messages.map { message -> [String: Any] in
            return [
                "id": message.id,
                "text": message.text,
                "senderId": message.senderId,
                "timestamp": Timestamp(date: message.timestamp),
                "isFromLister": message.isFromLister
            ]
        }
        
        var data: [String: Any] = [
            "name": name,
            "age": age,
            "occupation": occupation,
            "budget": budget,
            "moveInDate": Timestamp(date: moveInDate),
            "listingId": listingId,
            "dateInterested": Timestamp(date: dateInterested),
            "messages": messagesData,
            "status": status.rawValue,
            "initialMessage": initialMessage
        ]
        
        if let profileImage = profileImage {
            data["profileImage"] = profileImage
        }
        
        return data
    }
    
    static func sampleInterestedUsers() -> [InterestedUser] {
        return [
            InterestedUser(
                id: "user1",
                name: "Emma Johnson",
                age: 26,
                occupation: "Software Engineer",
                budget: 2500,
                moveInDate: Date().addingTimeInterval(86400 * 30), // 30 days from now
                profileImage: "profile1",
                initialMessage: "Hi, I love the apartment and would like to schedule a viewing. I work from home so the extra space would be perfect for my home office setup.",
                listingId: "1",
                dateInterested: Date().addingTimeInterval(-86400 * 3), // 3 days ago
                messages: [
                    Message(
                        id: "msg1",
                        text: "Hi, I love the apartment and would like to schedule a viewing. I work from home so the extra space would be perfect for my home office setup.",
                        senderId: "user1",
                        timestamp: Date().addingTimeInterval(-86400 * 3),
                        isFromLister: false
                    ),
                    Message(
                        id: "msg2",
                        text: "Hi Emma, thanks for your interest! I'd be happy to schedule a viewing. How does this Saturday at 2pm work for you?",
                        senderId: "owner1",
                        timestamp: Date().addingTimeInterval(-86400 * 2),
                        isFromLister: true
                    ),
                    Message(
                        id: "msg3",
                        text: "Saturday at 2pm works great for me! Looking forward to seeing the apartment.",
                        senderId: "user1",
                        timestamp: Date().addingTimeInterval(-86400 * 2 + 3600),
                        isFromLister: false
                    )
                ],
                status: .scheduled
            ),
            InterestedUser(
                id: "user2",
                name: "Michael Chen",
                age: 31,
                occupation: "Marketing Director",
                budget: 2300,
                moveInDate: Date().addingTimeInterval(86400 * 45), // 45 days from now
                profileImage: "profile2",
                initialMessage: "I'm interested in this apartment. Is there any flexibility on the rent? I'm looking for a 1-year lease.",
                listingId: "1",
                dateInterested: Date().addingTimeInterval(-86400 * 5), // 5 days ago
                messages: [
                    Message(
                        id: "msg4",
                        text: "I'm interested in this apartment. Is there any flexibility on the rent? I'm looking for a 1-year lease.",
                        senderId: "user2",
                        timestamp: Date().addingTimeInterval(-86400 * 5),
                        isFromLister: false
                    ),
                    Message(
                        id: "msg5",
                        text: "Hello Michael, thanks for your interest. We have some flexibility for a 1-year lease. Would you like to come see the apartment first?",
                        senderId: "owner1",
                        timestamp: Date().addingTimeInterval(-86400 * 4),
                        isFromLister: true
                    )
                ],
                status: .contacted
            ),
            InterestedUser(
                id: "user3",
                name: "Sophia Martinez",
                age: 28,
                occupation: "Graphic Designer",
                budget: 2100,
                moveInDate: Date().addingTimeInterval(86400 * 20), // 20 days from now
                profileImage: "profile3",
                initialMessage: "This looks perfect for my needs! I'm particularly interested in the rooftop deck. Is it available for all residents? Also, are utilities included in the rent?",
                listingId: "1",
                dateInterested: Date().addingTimeInterval(-86400 * 1), // 1 day ago
                messages: [
                    Message(
                        id: "msg6",
                        text: "This looks perfect for my needs! I'm particularly interested in the rooftop deck. Is it available for all residents? Also, are utilities included in the rent?",
                        senderId: "user3",
                        timestamp: Date().addingTimeInterval(-86400 * 1),
                        isFromLister: false
                    )
                ],
                status: .new
            ),
            InterestedUser(
                id: "user4",
                name: "James Wilson",
                age: 34,
                occupation: "Financial Analyst",
                budget: 3500,
                moveInDate: Date().addingTimeInterval(86400 * 14), // 14 days from now
                profileImage: "profile4",
                initialMessage: "I'm relocating to the area for work and this loft looks exactly what I'm looking for. I'd like to arrange a virtual tour if possible since I'm currently out of state.",
                listingId: "2",
                dateInterested: Date().addingTimeInterval(-86400 * 2), // 2 days ago
                messages: [
                    Message(
                        id: "msg7",
                        text: "I'm relocating to the area for work and this loft looks exactly what I'm looking for. I'd like to arrange a virtual tour if possible since I'm currently out of state.",
                        senderId: "user4",
                        timestamp: Date().addingTimeInterval(-86400 * 2),
                        isFromLister: false
                    ),
                    Message(
                        id: "msg8",
                        text: "Hi James, we'd be happy to arrange a virtual tour. How does tomorrow at 4pm Eastern time work for you?",
                        senderId: "owner1",
                        timestamp: Date().addingTimeInterval(-86400 * 1),
                        isFromLister: true
                    ),
                    Message(
                        id: "msg9",
                        text: "That works perfectly! Please send me the meeting link when you can. Looking forward to seeing the space.",
                        senderId: "user4",
                        timestamp: Date().addingTimeInterval(-86400 * 1 + 1800),
                        isFromLister: false
                    )
                ],
                status: .scheduled
            )
        ]
    }
}

// MARK: - Interest Status
enum InterestStatus: String, CaseIterable {
    case new = "New"
    case contacted = "Contacted"
    case scheduled = "Tour Scheduled"
    case applied = "Applied"
    case approved = "Approved"
    case rejected = "Rejected"
    case withdrawn = "Withdrawn"
    
    var color: Color {
        switch self {
        case .new: return .blue
        case .contacted: return .purple
        case .scheduled: return .orange
        case .applied: return .green
        case .approved: return .green
        case .rejected: return .red
        case .withdrawn: return .gray
        }
    }
    
    var icon: String {
        switch self {
        case .new: return "bell.fill"
        case .contacted: return "message.fill"
        case .scheduled: return "calendar.badge.clock"
        case .applied: return "doc.text.fill"
        case .approved: return "checkmark.circle.fill"
        case .rejected: return "xmark.circle.fill"
        case .withdrawn: return "arrow.uturn.backward.circle.fill"
        }
    }
}

// MARK: - Message
struct Message: Identifiable, Hashable {
    var id: String
    var text: String
    var senderId: String
    var timestamp: Date
    var isFromLister: Bool
    
    static func == (lhs: Message, rhs: Message) -> Bool {
        return lhs.id == rhs.id
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

// MARK: - Quick Reply
struct QuickReply: Identifiable {
    var id = UUID()
    var text: String
    
    static let common: [QuickReply] = [
        QuickReply(text: "Yes, the apartment is still available."),
        QuickReply(text: "When would you like to schedule a tour?"),
        QuickReply(text: "Utilities are not included in the rent."),
        QuickReply(text: "The security deposit is one month's rent."),
        QuickReply(text: "We require a credit check and proof of income."),
        QuickReply(text: "Yes, parking is available for an additional fee.")
    ]
}

// MARK: - Analytics Data
struct AnalyticsDataPoint: Identifiable {
    var id = UUID()
    var day: String
    var value: Int
}

enum TimeFrame: String, CaseIterable {
    case day = "Day"
    case week = "Week"
    case month = "Month"
    case year = "Year"
}

// MARK: - Simple Listing Info
struct SimpleListingInfo: Identifiable {
    var id: String
    var name: String
}

// MARK: - Filter Options
struct FilterOptions {
    var priceRange: ClosedRange<Double> = 0...5000
    var bedrooms: Int = 1
    var bathrooms: Int = 1
    var petPolicy: PetPolicy? = nil
    var leaseLength: LeaseLength? = nil
    var city: String = ""
    var moveInDate: Date = Date()
    var amenities: [String] = []
    var showActiveOnly: Bool = true
} 