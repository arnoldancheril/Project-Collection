//
//  PhotoUploadView.swift
//  RoommateSwipe
//

import SwiftUI
import PhotosUI

struct PhotoUploadView: View {
    @Binding var userData: UserRegistrationData
    @State private var showingImagePicker = false
    @State private var showingPropertyImagePicker = false
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedPropertyItems: [PhotosPickerItem] = []
    
    var body: some View {
        VStack(spacing: 25) {
            Text("Add Your Photos")
                .font(.title2)
                .fontWeight(.bold)
            
            ScrollView {
                VStack(spacing: 30) {
                    // Profile Photo
                    VStack(spacing: 15) {
                        Text("Profile Photo")
                            .fontWeight(.medium)
                        
                        if let profilePhoto = userData.profilePhoto {
                            Image(uiImage: profilePhoto)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 150, height: 150)
                                .clipShape(Circle())
                                .overlay(Circle().stroke(Color.blue, lineWidth: 2))
                        } else {
                            Image(systemName: "person.circle.fill")
                                .resizable()
                                .scaledToFit()
                                .frame(width: 150, height: 150)
                                .foregroundColor(.gray)
                        }
                        
                        PhotosPicker(
                            selection: $selectedItem,
                            matching: .images,
                            photoLibrary: .shared()) {
                                Text("Select Profile Photo")
                                    .foregroundColor(.blue)
                            }
                            .onChange(of: selectedItem) { newItem in
                                Task {
                                    if let data = try? await newItem?.loadTransferable(type: Data.self),
                                       let image = UIImage(data: data) {
                                        userData.profilePhoto = image
                                    }
                                }
                            }
                    }
                    
                    Divider()
                    
                    // Property Photos
                    VStack(spacing: 15) {
                        Text("Property Photos (Optional)")
                            .fontWeight(.medium)
                        
                        if !userData.propertyPhotos.isEmpty {
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 10) {
                                    ForEach(userData.propertyPhotos, id: \.self) { photo in
                                        Image(uiImage: photo)
                                            .resizable()
                                            .scaledToFill()
                                            .frame(width: 120, height: 120)
                                            .clipShape(RoundedRectangle(cornerRadius: 10))
                                    }
                                }
                                .padding(.horizontal)
                            }
                        }
                        
                        PhotosPicker(
                            selection: $selectedPropertyItems,
                            maxSelectionCount: 5,
                            matching: .images,
                            photoLibrary: .shared()) {
                                Label("Add Property Photos", systemImage: "photo.stack")
                                    .foregroundColor(.blue)
                            }
                            .onChange(of: selectedPropertyItems) { newItems in
                                Task {
                                    var images: [UIImage] = []
                                    for item in newItems {
                                        if let data = try? await item.loadTransferable(type: Data.self),
                                           let image = UIImage(data: data) {
                                            images.append(image)
                                        }
                                    }
                                    userData.propertyPhotos = images
                                }
                            }
                    }
                    
                    // Photo Guidelines
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Photo Guidelines:")
                            .fontWeight(.medium)
                        
                        VStack(alignment: .leading, spacing: 5) {
                            BulletPoint(text: "Use a clear, recent photo of yourself")
                            BulletPoint(text: "Ensure good lighting and quality")
                            BulletPoint(text: "Property photos should show the space accurately")
                            BulletPoint(text: "Avoid using filters or heavy editing")
                        }
                        .foregroundColor(.gray)
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
                }
                .padding()
            }
        }
    }
}

struct BulletPoint: View {
    let text: String
    
    var body: some View {
        HStack(alignment: .top) {
            Text("•")
            Text(text)
        }
    }
}

#Preview {
    PhotoUploadView(userData: .constant(UserRegistrationData()))
        .padding()
} 