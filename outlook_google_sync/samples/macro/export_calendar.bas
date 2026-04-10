Attribute VB_Name = "ExportCalendar"
Option Explicit

Sub ExportCalendarToIcs()
    Dim ns As Outlook.NameSpace
    Dim cal As Outlook.Folder
    Dim fso As Object
    Dim ts As Object
    Dim outPath As String
    Dim item As Object

    outPath = Environ("USERPROFILE") & "\Downloads\outlook_export.ics"

    Set ns = Application.GetNamespace("MAPI")
    Set cal = ns.GetDefaultFolder(olFolderCalendar)
    Set fso = CreateObject("Scripting.FileSystemObject")
    Set ts = fso.CreateTextFile(outPath, True, True)

    ts.WriteLine "BEGIN:VCALENDAR"
    ts.WriteLine "VERSION:2.0"
    ts.WriteLine "PRODID:-//OutlookExport//EN"

    For Each item In cal.Items
        If TypeName(item) = "AppointmentItem" Then
            ts.WriteLine "BEGIN:VEVENT"
            ts.WriteLine "UID:" & item.EntryID
            ts.WriteLine "DTSTART:" & Format(item.StartUTC, "yyyymmdd\THHmmss\Z")
            ts.WriteLine "DTEND:" & Format(item.EndUTC, "yyyymmdd\THHmmss\Z")
            ts.WriteLine "SUMMARY:" & Replace(item.Subject, vbCrLf, " ")
            ts.WriteLine "LOCATION:" & Replace(item.Location, vbCrLf, " ")
            ts.WriteLine "DESCRIPTION:" & Replace(item.Body, vbCrLf, " ")
            ts.WriteLine "END:VEVENT"
        End If
    Next

    ts.WriteLine "END:VCALENDAR"
    ts.Close

    MsgBox "Exported: " & outPath
End Sub
