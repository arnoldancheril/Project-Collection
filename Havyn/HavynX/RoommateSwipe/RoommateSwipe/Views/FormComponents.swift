//
//  FormComponents.swift
//  RoommateSwipe
//
//  Created by  on 3/25/25.
//

import SwiftUI

// Standard form field with title and text input
struct FormField: View {
    let title: String
    @Binding var text: String
    var placeholder: String
    var keyboardType: UIKeyboardType = .default
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            
            TextField(placeholder, text: $text)
                .keyboardType(keyboardType)
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
        }
    }
}

// Secure form field for passwords
struct SecureFormField: View {
    let title: String
    @Binding var text: String
    var placeholder: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            
            SecureField(placeholder, text: $text)
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
        }
    }
}

// Form section with title and content
struct FormSection<Content: View>: View {
    let title: String
    let content: Content
    
    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
                .foregroundColor(.primary)
            
            content
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(12)
        }
    }
} 