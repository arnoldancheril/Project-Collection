import SwiftUI

struct ApartmentListingSignUpView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var currentStep = 0
    @State private var userData = ApartmentListingRegistrationData()
    @State private var isLoading = false
    @State private var showAlert = false
    @State private var alertTitle = ""
    @State private var alertMessage = ""
    var onComplete: (() -> Void)?
    
    var body: some View {
        ZStack {
            // Background gradient
            LinearGradient(
                gradient: Gradient(colors: [Color.blue.opacity(0.1), Color.purple.opacity(0.1)]),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            VStack(spacing: 20) {
                // Progress bar
                ProgressBar(currentStep: currentStep, totalSteps: 4)
                    .padding(.horizontal)
                
                // Step title
                Text(stepTitle)
                    .font(.title2)
                    .fontWeight(.bold)
                    .padding(.top, 10)
                
                // Step content
                ScrollView {
                    VStack(spacing: 20) {
                        switch currentStep {
                        case 0:
                            CompanyInfoView(userData: $userData)
                                .transition(.opacity)
                        case 1:
                            PropertyDetailsView(userData: $userData)
                                .transition(.opacity)
                        case 2:
                            PropertyFeaturesView(userData: $userData)
                                .transition(.opacity)
                        case 3:
                            PropertyVisualsView(userData: $userData)
                                .transition(.opacity)
                        default:
                            EmptyView()
                        }
                    }
                    .padding()
                }
                
                Spacer()
                
                // Navigation buttons
                HStack(spacing: 20) {
                    if currentStep > 0 {
                        Button(action: {
                            withAnimation {
                                currentStep -= 1
                            }
                        }) {
                            Text("Back")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.gray.opacity(0.2))
                                .cornerRadius(12)
                        }
                    }
                    
                    Button(action: {
                        withAnimation {
                            if currentStep < 3 {
                                currentStep += 1
                            } else {
                                // Registration complete - save to Firebase
                                saveToFirebase()
                            }
                        }
                    }) {
                        Text(currentStep == 3 ? "Complete" : "Next")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(
                                LinearGradient(
                                    gradient: Gradient(colors: [Color.blue, Color.purple]),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                    .disabled(isLoading)
                }
                .padding()
            }
            
            // Loading overlay
            if isLoading {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()
                
                VStack {
                    ProgressView()
                        .scaleEffect(1.5)
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .padding()
                    
                    Text("Saving your listing...")
                        .foregroundColor(.white)
                        .font(.headline)
                }
                .padding(30)
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.gray.opacity(0.7))
                )
            }
        }
        .navigationBarTitle("Apartment Listing", displayMode: .inline)
        .navigationBarBackButtonHidden(true)
        .navigationBarItems(leading: Button("Cancel") {
            dismiss()
        })
        .alert(isPresented: $showAlert) {
            Alert(
                title: Text(alertTitle),
                message: Text(alertMessage),
                dismissButton: .default(Text("OK")) {
                    if alertTitle == "Success" {
                        onComplete?()
                    }
                }
            )
        }
    }
    
    private var stepTitle: String {
        switch currentStep {
        case 0:
            return "Company Information"
        case 1:
            return "Property Details"
        case 2:
            return "Property Features"
        case 3:
            return "Upload Visuals"
        default:
            return ""
        }
    }
    
    private func saveToFirebase() {
        isLoading = true
        
        // Create a unique listing ID
        let listingId = UUID().uuidString
        
        FirebaseService.shared.saveApartmentListing(userData, listingId: listingId) { listingId, error in
            isLoading = false
            
            if let error = error {
                alertTitle = "Error"
                alertMessage = "Failed to save your listing: \(error.localizedDescription)"
                showAlert = true
            } else {
                alertTitle = "Success"
                alertMessage = "Your apartment listing has been created successfully!"
                showAlert = true
                
                // If we had image storage set up, we would upload images here
                // For now, we're just recording that they exist
            }
        }
    }
}

// Step 1: Company Information
struct CompanyInfoView: View {
    @Binding var userData: ApartmentListingRegistrationData
    
    var body: some View {
        VStack(spacing: 20) {
            FormField(title: "Apartment Company Name", text: $userData.companyName, placeholder: "Enter company name")
            
            FormField(title: "Contact Person Name", text: $userData.contactPersonName, placeholder: "Enter contact name")
            
            FormField(title: "Email Address", text: $userData.email, placeholder: "Enter email address", keyboardType: .emailAddress)
            
            SecureFormField(title: "Password", text: $userData.password, placeholder: "Create a password")
            
            FormField(title: "Contact Phone Number", text: $userData.contactPhoneNumber, placeholder: "Enter phone number", keyboardType: .phonePad)
        }
    }
}

// Step 2: Property Details
struct PropertyDetailsView: View {
    @Binding var userData: ApartmentListingRegistrationData
    
    var body: some View {
        VStack(spacing: 20) {
            FormField(title: "City", text: $userData.city, placeholder: "Enter city")
            
            FormField(title: "Neighborhood", text: $userData.neighborhood, placeholder: "Enter neighborhood")
            
            FormField(title: "Apartment Name (if applicable)", text: $userData.apartmentName, placeholder: "Enter apartment name")
            
            FormField(title: "Address", text: $userData.address, placeholder: "Enter property address")
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Number of Available Rooms")
                    .font(.headline)
                
                Stepper("\(userData.numberOfAvailableRooms) Rooms", value: $userData.numberOfAvailableRooms, in: 1...50)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Availability Date")
                    .font(.headline)
                
                DatePicker("", selection: $userData.availabilityDate, displayedComponents: .date)
                    .datePickerStyle(WheelDatePickerStyle())
                    .labelsHidden()
                    .frame(maxHeight: 180)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Price Range ($)")
                    .font(.headline)
                
                HStack {
                    Text("$\(userData.minPrice) - $\(userData.maxPrice)")
                        .foregroundColor(.secondary)
                }
                .padding(.bottom, 8)
                
                Text("Minimum Price")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Slider(value: Binding(
                    get: { Double(userData.minPrice) },
                    set: { userData.minPrice = Int($0) }
                ), in: 300...5000, step: 50)
                .padding(.bottom, 8)
                
                Text("Maximum Price")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Slider(value: Binding(
                    get: { Double(userData.maxPrice) },
                    set: { userData.maxPrice = Int($0) }
                ), in: 300...10000, step: 50)
            }
            .padding()
            .background(Color(.secondarySystemBackground))
            .cornerRadius(10)
        }
    }
}

// Step 3: Property Features
struct PropertyFeaturesView: View {
    @Binding var userData: ApartmentListingRegistrationData
    
    private let housingTypes = ["Apartment", "Condo", "House"]
    private let amenitiesList = [
        "Pool", "Gym", "Parking", "Laundry", "Dishwasher", "Air Conditioning",
        "Heating", "Balcony", "Elevator", "Doorman", "Furnished", "Utilities Included",
        "Wi-Fi", "Cable TV", "Pets Allowed", "Wheelchair Access", "EV Charging", "Storage"
    ]
    
    var body: some View {
        VStack(spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Housing Type")
                    .font(.headline)
                
                Picker("", selection: $userData.housingType) {
                    ForEach(housingTypes, id: \.self) { type in
                        Text(type).tag(type)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 12) {
                Text("Amenities")
                    .font(.headline)
                
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 150))], spacing: 10) {
                        ForEach(amenitiesList, id: \.self) { amenity in
                            AmenityTag(
                                amenity: amenity,
                                isSelected: userData.amenities.contains(amenity),
                                action: {
                                    if userData.amenities.contains(amenity) {
                                        userData.amenities.removeAll { $0 == amenity }
                                    } else {
                                        userData.amenities.append(amenity)
                                    }
                                }
                            )
                        }
                    }
                }
                .frame(height: 200)
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Pet Policy")
                    .font(.headline)
                
                Toggle("Pet Friendly", isOn: $userData.isPetFriendly)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Smoking Policy")
                    .font(.headline)
                
                Toggle("Smoking Allowed", isOn: $userData.isSmokingAllowed)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
            }
        }
    }
}

// Step 4: Upload Visuals
struct PropertyVisualsView: View {
    @Binding var userData: ApartmentListingRegistrationData
    
    var body: some View {
        VStack(spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Property Description")
                    .font(.headline)
                
                TextEditor(text: $userData.propertyDescription)
                    .frame(height: 150)
                    .padding(4)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                    )
            }
            
            VStack(alignment: .leading, spacing: 12) {
                Text("Upload Property Images")
                    .font(.headline)
                
                VStack(spacing: 10) {
                    Button(action: {
                        // Photo picker would go here - not implementing for this mockup
                    }) {
                        HStack {
                            Image(systemName: "photo.on.rectangle")
                            Text("Choose Photos")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .foregroundColor(.blue)
                        .cornerRadius(10)
                    }
                    
                    Text("Upload high-quality images to showcase your property")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                    
                    // Placeholder for image grid
                    if userData.propertyPhotos.isEmpty {
                        ZStack {
                            RoundedRectangle(cornerRadius: 10)
                                .fill(Color.gray.opacity(0.1))
                            
                            VStack {
                                Image(systemName: "photo.on.rectangle.angled")
                                    .font(.system(size: 40))
                                    .foregroundColor(.gray)
                                Text("No photos added yet")
                                    .foregroundColor(.gray)
                            }
                        }
                        .frame(height: 200)
                    }
                }
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
            }
        }
    }
}

// Amenity Tag component
struct AmenityTag: View {
    let amenity: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .bold))
                }
                
                Text(amenity)
                    .font(.system(size: 14))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity)
            .background(isSelected ? Color.blue : Color.gray.opacity(0.2))
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(8)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct ApartmentListingSignUpView_Previews: PreviewProvider {
    static var previews: some View {
        ApartmentListingSignUpView()
    }
} 