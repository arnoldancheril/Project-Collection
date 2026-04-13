//
//  ListingEditView.swift
//  RoommateSwipe
//

import SwiftUI

struct ListingEditView: View {
    let listing: ApartmentListing
    let onSave: (ApartmentListing) -> Void
    
    @State private var name: String
    @State private var address: String
    @State private var city: String
    @State private var state: String
    @State private var zipCode: String
    @State private var description: String
    @State private var monthlyRent: Double
    @State private var bedrooms: Int
    @State private var bathrooms: Double
    @State private var squareFootage: Int
    @State private var availableDate: Date
    @State private var selectedLeaseLengths: [LeaseLength]
    @State private var selectedPetPolicy: PetPolicy
    @State private var isActive: Bool
    
    init(listing: ApartmentListing, onSave: @escaping (ApartmentListing) -> Void) {
        self.listing = listing
        self.onSave = onSave
        
        _name = State(initialValue: listing.name)
        _address = State(initialValue: listing.address)
        _city = State(initialValue: listing.city)
        _state = State(initialValue: listing.state)
        _zipCode = State(initialValue: listing.zipCode)
        _description = State(initialValue: listing.description)
        _monthlyRent = State(initialValue: listing.monthlyRent)
        _bedrooms = State(initialValue: listing.bedrooms)
        _bathrooms = State(initialValue: listing.bathrooms)
        _squareFootage = State(initialValue: listing.squareFootage)
        _availableDate = State(initialValue: listing.availableDate)
        _selectedLeaseLengths = State(initialValue: listing.leaseLength)
        _selectedPetPolicy = State(initialValue: listing.petPolicy)
        _isActive = State(initialValue: listing.isActive)
    }
    
    var body: some View {
        Form {
            Section(header: Text("BASIC INFORMATION")) {
                TextField("Property Name", text: $name)
                
                VStack(alignment: .leading) {
                    Text("Status")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Toggle(isOn: $isActive) {
                        Text(isActive ? "Active" : "Inactive")
                    }
                }
            }
            
            Section(header: Text("LOCATION")) {
                TextField("Address", text: $address)
                TextField("City", text: $city)
                TextField("State", text: $state)
                TextField("Zip Code", text: $zipCode)
            }
            
            Section(header: Text("DETAILS")) {
                VStack(alignment: .leading) {
                    Text("Monthly Rent")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    HStack {
                        Text("$")
                        TextField("Amount", value: $monthlyRent, formatter: NumberFormatter())
                            .keyboardType(.numberPad)
                    }
                }
                
                Stepper("Bedrooms: \(bedrooms)", value: $bedrooms, in: 1...10)
                
                Picker("Bathrooms", selection: $bathrooms) {
                    Text("1").tag(1.0)
                    Text("1.5").tag(1.5)
                    Text("2").tag(2.0)
                    Text("2.5").tag(2.5)
                    Text("3").tag(3.0)
                    Text("3.5").tag(3.5)
                    Text("4+").tag(4.0)
                }
                
                VStack(alignment: .leading) {
                    Text("Square Footage")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    TextField("Square Feet", value: $squareFootage, formatter: NumberFormatter())
                        .keyboardType(.numberPad)
                }
                
                DatePicker("Available From", selection: $availableDate, displayedComponents: .date)
            }
            
            Section(header: Text("LEASING OPTIONS")) {
                VStack(alignment: .leading) {
                    Text("Lease Lengths")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    ForEach(LeaseLength.allCases, id: \.self) { leaseLength in
                        Toggle(leaseLength.rawValue, isOn: Binding(
                            get: { selectedLeaseLengths.contains(leaseLength) },
                            set: { isSelected in
                                if isSelected {
                                    selectedLeaseLengths.append(leaseLength)
                                } else {
                                    selectedLeaseLengths.removeAll { $0 == leaseLength }
                                }
                            }
                        ))
                    }
                }
                
                Picker("Pet Policy", selection: $selectedPetPolicy) {
                    ForEach(PetPolicy.allCases, id: \.self) { policy in
                        Text(policy.rawValue).tag(policy)
                    }
                }
            }
            
            Section(header: Text("DESCRIPTION")) {
                TextEditor(text: $description)
                    .frame(height: 150)
            }
            
            Section {
                Button("Save Changes") {
                    let updatedListing = ApartmentListing(
                        id: listing.id,
                        ownerId: listing.ownerId,
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
                        availableDate: availableDate,
                        leaseLength: selectedLeaseLengths,
                        amenities: listing.amenities,
                        petPolicy: selectedPetPolicy,
                        images: listing.images,
                        coordinates: listing.coordinates,
                        isActive: isActive,
                        dateCreated: listing.dateCreated,
                        dateModified: Date(),
                        interestedUsers: listing.interestedUsers
                    )
                    
                    onSave(updatedListing)
                }
                .frame(maxWidth: .infinity, alignment: .center)
                .foregroundColor(.blue)
            }
        }
    }
}

#Preview {
    ListingEditView(listing: ApartmentListing.sampleListings()[0]) { _ in }
} 