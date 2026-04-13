//
//  ScheduleView.swift
//  RoommateSwipe
//

import SwiftUI

struct ScheduleView: View {
    @Binding var userData: UserRegistrationData
    
    var body: some View {
        VStack(spacing: 25) {
            Text("Your Schedule")
                .font(.title2)
                .fontWeight(.bold)
            
            ScrollView {
                VStack(alignment: .leading, spacing: 25) {
                    // Wake up time
                    VStack(alignment: .leading) {
                        Text("What time do you usually wake up?")
                            .fontWeight(.medium)
                        
                        DatePicker("Wake up time",
                                 selection: $userData.wakeUpTime,
                                 displayedComponents: .hourAndMinute)
                            .datePickerStyle(.wheel)
                            .labelsHidden()
                    }
                    
                    // Bedtime
                    VStack(alignment: .leading) {
                        Text("What time do you usually go to bed?")
                            .fontWeight(.medium)
                        
                        DatePicker("Bedtime",
                                 selection: $userData.bedTime,
                                 displayedComponents: .hourAndMinute)
                            .datePickerStyle(.wheel)
                            .labelsHidden()
                    }
                    
                    // Work Schedule
                    VStack(alignment: .leading) {
                        Text("What's your typical work/study schedule?")
                            .fontWeight(.medium)
                        
                        Picker("Work Schedule", selection: $userData.workSchedule) {
                            Text("9-5").tag("9-5")
                            Text("Night Shift").tag("Night Shift")
                            Text("Flexible").tag("Flexible")
                            Text("Remote").tag("Remote")
                            Text("Student").tag("Student")
                            Text("Other").tag("Other")
                        }
                        .pickerStyle(.wheel)
                    }
                    
                    // Occupation
                    VStack(alignment: .leading) {
                        Text("What's your occupation?")
                            .fontWeight(.medium)
                        
                        TextField("e.g. Software Engineer, Student, etc.", text: $userData.occupation)
                            .textFieldStyle(.roundedBorder)
                    }
                    
                    // Additional Info
                    VStack(alignment: .leading) {
                        Text("Any additional schedule information?")
                            .fontWeight(.medium)
                        
                        TextEditor(text: .constant(""))
                            .frame(height: 100)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                            )
                    }
                }
                .padding()
            }
        }
    }
}

#Preview {
    ScheduleView(userData: .constant(UserRegistrationData()))
        .padding()
} 